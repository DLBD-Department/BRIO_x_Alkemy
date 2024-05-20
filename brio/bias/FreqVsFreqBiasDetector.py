from .BiasDetector import BiasDetector
from .threshold_calculator import threshold_calculator
from sklearn.utils.extmath import cartesian
from itertools import combinations
from scipy.spatial.distance import jensenshannon
import numpy as np

class FreqVsFreqBiasDetector(BiasDetector):

    def __init__(self, distance: str, aggregating_function=max, A1="high", target_variable_type='class'):
        '''
            distance: which distance will be used to compute the bias detection
            aggregating_function: function needed to aggregate distances for multi-class comparisons
            A1: sensitivity parameter used to computer the parametric threshold
            target_variable_type: type of the tgt variable. 'class' or 'probability'
        '''
        self.dis = distance
        self.aggregating_function = aggregating_function
        self.A1 = A1
        self.target_variable_type = target_variable_type


    def compute_distance_between_frequencies(self, 
            observed_distribution):
        '''
        observed_distribution: list of numpy arrays, 
            each of them containing the distribution target_variable | root_variable. 
            e.g. [ array(female_0, female_1), array(male_0, male_1) ]  
            The lenght of the list is given by the number of categories of the root variable.
            The shape of each array is given by the number of labels of target_variable (if self.target_variable_type=='class') 
            or the number of bins (if self.target_variable_type=='probability').
            
        It works for any number of labels of the target variable and any number of classes for the root variable. 
        The final distance is given by self.aggregating_function. 
        '''

        distances = []

        if self.dis == "TVD":
            for pair in combinations(observed_distribution, 2):
                # TVD
                distance = max( abs( pair[0]-pair[1] ) )
                distances.append(distance)            
        elif self.dis == "JS":
            for pair in combinations(observed_distribution, 2):
                # Squaring JS given that the scipy implementation has square root
                distance = jensenshannon(p=pair[0], q=pair[1], base=2)**2
                # If no observation are present for one class, the JS distance will be a nan.
                # Changing into None to keep the functionalities of Risk Measurement. 
                if np.isnan(distance):
                    distances.append(None)
                else:
                    distances.append(distance)
        else:
            raise Exception("Only TVD or JS are supported as distances for freq_vs_freq analysis")

        #TODO should we use any or all? For binary classes it's the same,
        #but for multi-classes it's not.
        #Currently, if for any class the distance is undefined (None), we return 
        #an overall None distance and a None standard deviation. 
        if any(d is None for d in distances):
            overall_distance = None
            return overall_distance, None
        else:
            overall_distance = self.aggregating_function(distances)
            if len(distances) > 1:
                ## Computing the standard deviation of the distances in case of 
                # multi-class root_variable
                return overall_distance, np.std(distances)
            else:
                return overall_distance, None


    def compare_root_variable_groups(self,
            dataframe,
            target_variable,
            root_variable,
            threshold=None,
            n_bins=10):
        '''
        This function compares for the two groups given by
        the two categories of root_variable and check if 
        the observed frequencies of a given target_variable
        are distant below the given threshold.

        Args:
            dataframe: Pandas DataFrame with features and 
                predictions
            target_variable: variable with the predictions (either prediction lables or probabilities, see self.target_variable_type)
            root_variable: variable that we use to compare the predicted
                labels in two different groups of observation
            threshold: value from 0 to 1 used to check the computed distance with. If None, the tool will compute a parametric threshold. 
            n_bins: number of bins used to split the predicted probability (only applies when self.target_variable_type='probability')

        Returns:
            A tuple (distance, distance<=computed_threshold, computed_threshold, standard_deviation).
        '''

        root_variable_labels = sorted(dataframe[root_variable].unique())
        A2 = len(root_variable_labels)

        if self.target_variable_type == 'class':
            target_variable_labels = sorted(dataframe[target_variable].unique())
            freqs, abs_freqs = self.get_frequencies_list(
                                dataframe,
                                target_variable,
                                target_variable_labels,
                                root_variable,
                                root_variable_labels) 
        elif self.target_variable_type == 'probability':
            target_variable_labels = None
            freqs, abs_freqs = self.get_frequencies_list_from_probs(
                                dataframe,
                                target_variable,
                                root_variable,
                                root_variable_labels,
                                n_bins) 
        else:
            raise Exception("target_variable_type can only be 'class' or 'probability'")
                    
        A3 = sum(sum(abs_freqs))
        computed_threshold = threshold_calculator(self.A1, A2, A3, default_threshold=threshold)
        distance, stds = self.compute_distance_between_frequencies(freqs)
        return (distance, distance<=computed_threshold, computed_threshold, stds)


    def compare_root_variable_conditioned_groups(self, 
            dataframe,
            target_variable,
            root_variable,
            conditioning_variables,
            threshold=None,
            min_obs_per_group=30,
            n_bins=10):
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
            threshold: value from 0 to 1 used to check the computed distance with. If None, the tool will compute a parametric threshold. 
            min_obs_per_group: the minimum number of observations needed for the distance computation
            n_bins: number of bins used to split the predicted probability (only applies when self.target_variable_type='probability')

        Returns:
            A dictionary {group_condition: (numb_obs_of_group, distance, distance>=computed_threshold, computed_threshold, standard_deviation)}
        '''

        # this is computed once and passed each time for each group
        # in order to avoid disappearing labels due to small groups
        # with only one observed category.
        root_variable_labels = sorted(dataframe[root_variable].unique())

        if self.target_variable_type == 'class':
            target_variable_labels = sorted(dataframe[target_variable].unique())
        elif self.target_variable_type == 'probability':
            target_variable_labels = None
        else:
            raise Exception("target_variable_type can only be 'class' or 'probability'")
        
        # Second parameter for threshold calculator
        A2 = len(root_variable_labels)
        conditioning_variables_subsets = list(self.powerset(conditioning_variables))

        # All the possible subsets of conditioning variables are inspected. The first one
        # is excluded being the empty set. 
        conditioned_frequencies = {}
        for conditioning_variables_subset in conditioning_variables_subsets[1:]:

            combinations = cartesian([dataframe[v].unique() for v in conditioning_variables_subset])

            for comb in combinations:
                condition = " & ".join(
                    [f'{conditioning_variables_subset[i[0]]}=={i[1]}' for i in enumerate(comb)]
                )

                dataframe_subset = dataframe.query(condition)
                num_of_obs = dataframe_subset.shape[0]

                if self.target_variable_type == 'class':
                    conditioned_frequencies[condition] = (
                            num_of_obs, 
                            self.get_frequencies_list(
                                dataframe_subset,
                                target_variable,
                                target_variable_labels,
                                root_variable,
                                root_variable_labels)[0] #taking the relative freqs, the absolute freqs are not needed here
                            )
                elif self.target_variable_type == 'probability':
                    conditioned_frequencies[condition] = (
                            num_of_obs, 
                            self.get_frequencies_list_from_probs(
                                dataframe_subset,
                                target_variable,
                                root_variable,
                                root_variable_labels,
                                n_bins)[0] #taking the relative freqs, the absolute freqs are not needed here
                            )
            
        distances = {
                # group: (number_of_observations, (overall_distance, standard_deviations) )
                group: (
                        (obs_and_freqs[0], 
                            self.compute_distance_between_frequencies(obs_and_freqs[1]) # (distance, standard_deviations)
                        )
                    ) for group, obs_and_freqs in conditioned_frequencies.items()
                }
        
        results = {}
        for group, obs_and_dist in distances.items():
            # Too small groups
            if obs_and_dist[0] < min_obs_per_group:
                result = (obs_and_dist[0], None, 'Not enough observations')
            # Groups for which distance is not defined (only one class available, JS computed)
            elif obs_and_dist[1][0] is None:
                result = (obs_and_dist[0], None, 'Distance non defined')
            else:
                result = (
                    obs_and_dist[0], #This will also be the A3 for threshold_calculator, being it the number of obs of the group
                    obs_and_dist[1][0], #distance
                    obs_and_dist[1][0]<=threshold_calculator(A1=self.A1, A2=A2, A3=obs_and_dist[0], default_threshold=threshold),
                    threshold_calculator(A1=self.A1, A2=A2, A3=obs_and_dist[0], default_threshold=threshold),
                    obs_and_dist[1][1] #standard deviation
                )
            results[group] = result
            
        return results