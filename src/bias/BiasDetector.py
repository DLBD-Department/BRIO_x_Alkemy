import numpy as np
from sklearn.utils.extmath import cartesian
from itertools import chain
from itertools import combinations as itertools_combinations

from .threshold_calculator import threshold_calculator

class BiasDetector:

    def __init__(self, distance, A1="high"):
        '''
            distance: which distance will be used to compute the bias detection
            A1: sensitivity parameter used to computer the parametric threshold
        '''
        self.dis = distance
        self.A1 = A1


    def powerset(self, iterable):
        '''
        Recipe from itertool documentation. 
        https://docs.python.org/3/library/itertools.html#itertools-recipes
        '''
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        return chain.from_iterable(itertools_combinations(s, r) for r in range(len(s)+1))


    def get_frequencies_list(self,
            dataframe,
            target_variable,
            target_variable_labels,
            root_variable,
            root_variable_labels):
        '''
        This function builds a list of numpy arrays, 
        each of them containing the distribution target_variable | root_variable. 
        e.g. [ array(female_0, female_1), array(male_0, male_1) ] 

        The lenght of the list is given by the number of categories of the root variable.
        The shape of each array is given by the number of labels of target_variable.
        '''

        #TODO cosa succede se non tutte le labels di root_variable sono presenti nel dataframe?
        freq_list = []
        abs_freq_list = []
        for label in root_variable_labels:
            dataframe_subset = dataframe.loc[
                dataframe[root_variable]==label
            ]

            freq_list.append(
                np.array(
                    dataframe_subset[target_variable].value_counts(normalize=True)
                    .reindex(target_variable_labels, fill_value=0)
                )
            )

            abs_freq_list.append(
                np.array(
                    dataframe_subset[target_variable].value_counts(normalize=False)
                    .reindex(target_variable_labels, fill_value=0)
                )
            )

        return freq_list, abs_freq_list
    

    def compare_root_variable_groups(self,
            dataframe,
            target_variable,
            root_variable,
            threshold=None,
            reference_distribution=None):
        '''
        This function compares for the two groups given by
        the two categories of root_variable and check if 
        the observed frequencies of a given target_variable
        are distant below the given threshold
        Distance from a reference distribution is computed if
        a reference_distribution is passed. 
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
        if reference_distribution is None:
            distance = self.dis.compute_distance_between_frequencies(freqs)
            return (distance, distance<=computed_threshold, computed_threshold)
        else:
            distance = self.dis.compute_distance_from_reference(freqs, reference_distribution)
            return (distance, [d<=computed_threshold for d in distance], computed_threshold)


    def compare_root_variable_conditioned_groups(self, 
            dataframe,
            target_variable,
            root_variable,
            conditioning_variables,
            threshold=None,
            min_obs_per_group=30,
            reference_distribution=None):
        '''
        This functions compares the distance between the two 
        categories of root_variable as observed in the two
        partitions of dataframe, partitions provided by the
        target_variable, conditioning it
        to each category of each conditioning_variable. The distance 
        is computed and compared to the given threshold.
        Distance from a reference distribution is computed if 
        reference_distribution is passed. 

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
            reference_distribution: numpy array of probabilities w.r.t. target_variable (e.g. predictions) for the 
                two groups given by root_variable. If provided, the distance from reference distribution is 
                computed. If not provided, the distance between probabilities is computed instead. 

        Returns:
            if reference_distribution is None, a dictionary {group_condition: (numb_obs_of_group, distance, distance>=threshold)}
            if reference_distribution is given, a dictionary {group_condition: (numb_obs_of_group, [distance_a, distance_b], [distance_a>=threshold, distance_b>=threshold])},
            where distance_a and distance_b are the distances computed for the two categories of root_variable. 
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
            
        if reference_distribution is None:
            distances = {
                    group: (
                            (obs_and_freqs[0], 
                             self.dis.compute_distance_between_frequencies(obs_and_freqs[1])
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
        else:
            distances = {
                    group: (
                        (
                            obs_and_freqs[0], 
                            self.dis.compute_distance_from_reference(obs_and_freqs[1], reference_distribution)
                        ) if obs_and_freqs[1] is not None else (obs_and_freqs[0], None)
                    ) for group, obs_and_freqs in conditioned_frequencies.items()
                }
    
            results = {group: (
                    (
                        obs_and_dist[0], 
                        obs_and_dist[1], 
                        [d<=threshold_calculator(A1=self.A1, A2=A2, A3=obs_and_dist[0], default_threshold=threshold) for d in obs_and_dist[1]]
                    ) if obs_and_dist[1] is not None else (obs_and_dist[0], obs_and_dist[1], 'Not enough observations')
                ) for group, obs_and_dist in distances.items()}
    
        return results
