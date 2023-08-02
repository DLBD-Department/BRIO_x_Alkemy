from .BiasDetector import BiasDetector
from .threshold_calculator import threshold_calculator
from sklearn.utils.extmath import cartesian
from itertools import combinations
from scipy.spatial.distance import jensenshannon

class FreqVsFreqBiasDetector(BiasDetector):

    def __init__(self, distance: str, aggregating_function=max, A1="high"):
        '''
            distance: which distance will be used to compute the bias detection
            aggregating_function: function needed to aggregate distances for multi-class comparisons
            A1: sensitivity parameter used to computer the parametric threshold
        '''
        self.dis = distance
        self.aggregating_function = aggregating_function
        self.A1 = A1


    def compute_distance_between_frequencies(self, 
            observed_distribution):
        '''
        observed_distribution: list of numpy arrays, 
            each of them containing the distribution target_variable | root_variable. 
            e.g. [ array(female_0, female_1), array(male_0, male_1) ]  
            The lenght of the list is given by the number of categories of the root variable.
            The shape of each array is given by the number of labels of target_variable.
            
        It works for any number of labels of the target variable and any number of classes for the root variable. 
        The final distance is given by self.aggregating_function. 

        It breaks if aggregating_function=stdev when the root_variable is binary (we would have a stdev
        of a single number)
        '''

        if self.dis == "TVD":
            # Computing the TVD for each pair of distributions
            distance = self.aggregating_function(
                    # TVD
                    [max( abs( pair[0]-pair[1] ) ) for pair in combinations(observed_distribution, 2)]
                )
        elif self.dis == "JS":
            divergences = []
            for pair in combinations(observed_distribution, 2):
                # Squaring JS given that the scipy implementation has square root
                divergence = jensenshannon(p=pair[0], q=pair[1], base=2)**2
                divergences.append(divergence)

            distance = self.aggregating_function(divergences)
        else:
            raise Exception("Only TVD or JS are supported as distances for freq_vs_freq analysis")

        return distance


    def compare_root_variable_groups(self,
            dataframe,
            target_variable,
            root_variable,
            threshold=None):
        '''
        This function compares for the two groups given by
        the two categories of root_variable and check if 
        the observed frequencies of a given target_variable
        are distant below the given threshold
        '''

        root_variable_labels = dataframe[root_variable].unique()
        A2 = len(root_variable_labels)
        target_variable_labels = dataframe[target_variable].unique()

        freqs, abs_freqs = self.get_frequencies_list(
                            dataframe,
                            target_variable,
                            target_variable_labels,
                            root_variable,
                            root_variable_labels) 
        
        A3 = sum(sum(abs_freqs))
        computed_threshold = threshold_calculator(self.A1, A2, A3, default_threshold=threshold)
        distance = self.compute_distance_between_frequencies(freqs)
        return (distance, distance<=computed_threshold, computed_threshold)


    def compare_root_variable_conditioned_groups(self, 
            dataframe,
            target_variable,
            root_variable,
            conditioning_variables,
            threshold=None,
            min_obs_per_group=30):
        '''
        This functions compares the distance between the two categories of root_variable as observed in the two
        partitions of dataframe, partitions provided by the target_variable, conditioning it
        to each category of each conditioning_variable. The distance is computed and compared to the given threshold.

        Args:
            dataframe: Pandas DataFrame with features and 
                predicted labels
            target_variable: variable with the predicted labels
            root_variable: variable that we use to compare the predicted
                labels in two different groups of observation
            conditioning_variables: list of strings names of the variables that we use 
                to create groups within the population. 
                Starting from the first variable, a tree of conditions is created;
                for each of the resulting group, we check if the predicted labels frequencies are significantly 
                different in the two groups of root_variable.
            threshold: value from 0 to 1 used to check the computed distance with.
            min_obs_per_group: the minimum number of observations needed for the distance computation

        Returns:
            A dictionary {group_condition: (numb_obs_of_group, distance, distance>=threshold)}
        '''

        # this is computed once and passed each time for each group
        # in order to avoid disappearing labels due to small groups
        # with only one observed category.
        root_variable_labels = dataframe[root_variable].unique()
        target_variable_labels = dataframe[target_variable].unique()

        # Second parameter for threshold calculator
        A2 = len(root_variable_labels)

        conditioned_frequencies = {}

        conditioning_variables_subsets = list(self.powerset(conditioning_variables))

        # All the possible subsets of conditioning variables are inspected. The first one
        # is excluded being the empty set. 
        for conditioning_variables_subset in conditioning_variables_subsets[1:]:

            combinations = cartesian([dataframe[v].unique() for v in conditioning_variables_subset])

            for comb in combinations:
                condition = " & ".join(
                    [f'{conditioning_variables_subset[i[0]]}=={i[1]}' for i in enumerate(comb)]
                )

                dataframe_subset = dataframe.query(condition)
                num_of_obs = dataframe_subset.shape[0]

                if num_of_obs >= min_obs_per_group:
                    conditioned_frequencies[condition] = (
                            num_of_obs, 
                            self.get_frequencies_list(
                                dataframe_subset,
                                target_variable,
                                target_variable_labels,
                                root_variable,
                                root_variable_labels)[0] #taking the relative freqs, the absolute freqs are not needed here
                            )
                else:
                    conditioned_frequencies[condition] = (num_of_obs, None)
            
        distances = {
                group: (
                        (obs_and_freqs[0], 
                            self.compute_distance_between_frequencies(obs_and_freqs[1])
                        ) if obs_and_freqs[1] is not None else (obs_and_freqs[0], None)
                    ) for group, obs_and_freqs in conditioned_frequencies.items()
                }

        results = {group: (
                (
                    obs_and_dist[0], #This will also be the A3 for threshold_calculator, being it the number of obs of the group
                    obs_and_dist[1], 
                    obs_and_dist[1]<=threshold_calculator(A1=self.A1, A2=A2, A3=obs_and_dist[0], default_threshold=threshold)
                ) if obs_and_dist[1] is not None else (obs_and_dist[0], obs_and_dist[1], 'Not enough observations')
            ) for group, obs_and_dist in distances.items()}
    
        return results