from .Distance import Distance
import numpy as np
from sklearn.utils.extmath import cartesian

class BiasDetector:

    def __init__(self):
        self.dis = Distance() 

    def get_frequencies_list(self,
            dataframe,
            labels_variable,
            binary_variable,
            binary_variable_labels):
        '''
        This function builds a list of numpy arrays, 
        each of them containing the observed frequencies of the
        two labels of labels_variable within the two groups
        given by the two categories of binary_variable. 
        '''

        predicted_ones = dataframe.loc[
            dataframe[labels_variable]==1
        ]
        
        predicted_zeros = dataframe.loc[
            dataframe[labels_variable]==0
        ]

        obs_freq_within_zeros = np.array(
            predicted_zeros[binary_variable].value_counts(normalize=True)
                .reindex(binary_variable_labels, fill_value=0)
        )

        obs_freq_within_ones= np.array(
            predicted_ones[binary_variable].value_counts(normalize=True)
                .reindex(binary_variable_labels, fill_value=0)
        )

        freq_list = [ 
            np.array([obs_freq_within_zeros[0], obs_freq_within_zeros[1]]), 
            np.array([obs_freq_within_ones[0], obs_freq_within_ones[1]]) 
        ]

        return freq_list

    
    def compare_binary_variable_groups(self,
            dataframe,
            labels_variable,
            binary_variable,
            threshold,
            reference_distribution=None):
        '''
        This function compares for the two groups given by
        the two categories of binary_variable and check if 
        the observed frequencies of a given labels_variable
        are distant below the given threshold
        Distance from a reference distribution is computed if
        a reference_distribution is passed. 
        '''

        binary_variable_labels = dataframe[binary_variable].unique()

        freqs = self.get_frequencies_list(
                            dataframe,
                            labels_variable,
                            binary_variable,
                            binary_variable_labels) 
        
        if reference_distribution is None:
            distance = self.dis.compute_distance_between_frequencies(freqs)
            return (distance, distance<=threshold)
        else:
            distance = self.dis.compute_distance_from_reference(freqs, reference_distribution)
            return (distance, [d<=threshold for d in distance])


    def compare_binary_variable_conditioned_groups(self, 
            dataframe,
            labels_variable,
            binary_variable,
            conditioning_variables,
            threshold,
            min_obs_per_group=30,
            reference_distribution=None):
        '''
        This functions compares the distance between the two 
        categories of binary_variable as observed in the two
        partitions of dataframe, partitions provided by the
        labels_variable, conditioning it
        to each category of each conditioning_variable. The distance 
        is computed and compared to the given threshold.
        Distance from a reference distribution is computed if 
        reference_distribution is passed. 

        Args:
            dataframe: Pandas DataFrame with features and 
                predicted labels
            labels_variable: variable with the predicted labels
            binary_variable: variable that we use to compare the predicted
                labels in two different groups of observation
            conditioning_variables: list of strings names of the variables that we use 
                to create groups within the population. 
                Starting from the first variable, a tree of conditions is created;
                for each of the resulting group, we check if the predicted labels frequencies are significantly 
                different in the two groups of binary_variable.
            threshold: value from 0 to 1 used to check the computed distance with.
            min_obs_per_group: the minimum number of observations needed for the distance computation
            reference_distribution: numpy array of probabilities w.r.t. labels_variable (e.g. predictions) for the 
                two groups given by binary_variable. If provided, the distance from reference distribution is 
                computed. If not provided, the distance between probabilities is computed instead. 
        '''

        # this is computed once and passed each time for each group
        # in order to avoid disappearing labels due to small groups
        # with only one observed category.
        binary_variable_labels = dataframe[binary_variable].unique()
        conditioned_frequencies = {}
        var_index = 1

        # The conditioning variables are used to condition the dataset in a tree-based way:
        # the first variable is used, and for each of its categories a further conditioning 
        # is performed using the subsequent variables. 
        while var_index <= len(conditioning_variables):
            conditioning_variables_subset = conditioning_variables[:var_index]

            combinations = cartesian([dataframe[v].unique() for v in conditioning_variables_subset])

            for comb in combinations:
                condition = " & ".join(
                    [f'{conditioning_variables[i[0]]}=={i[1]}' for i in enumerate(comb)]
                )

                dataframe_subset = dataframe.query(condition)
                num_of_obs = dataframe_subset.shape[0]

                if num_of_obs >= min_obs_per_group:
                    conditioned_frequencies[condition] = (
                            num_of_obs, 
                            self.get_frequencies_list(
                                dataframe_subset,
                                labels_variable,
                                binary_variable,
                                binary_variable_labels)
                            )
                else:
                    conditioned_frequencies[condition] = (num_of_obs, None)

            var_index += 1
            
        if reference_distribution is None:
            distances = {
                    group: (
                        (obs_and_freqs[0], self.dis.compute_distance_between_frequencies(obs_and_freqs[1])) if obs_and_freqs[1] is not None else (obs_and_freqs[0], None)
                        ) for group, obs_and_freqs in conditioned_frequencies.items()
                    }
    
            results = {group: (
                (obs_and_dist[0], obs_and_dist[1], obs_and_dist[1]<=threshold) if obs_and_dist[1] is not None else (obs_and_dist[0], obs_and_dist[1], 'Not enough observations')
                ) for group, obs_and_dist in distances.items()}
        else:
            distances = {
                    group: (
                        (obs_and_freqs[0], self.dis.compute_distance_from_reference(obs_and_freqs[1], reference_distribution)) if obs_and_freqs[1] is not None else (obs_and_freqs[0], None)
                        ) for group, obs_and_freqs in conditioned_frequencies.items()
                    }
    
            results = {group: (
                (obs_and_dist[0], obs_and_dist[1], [d<=threshold for d in obs_and_dist[1]]) if obs_and_dist[1] is not None else (obs_and_dist[0], obs_and_dist[1], 'Not enough observations')
                ) for group, obs_and_dist in distances.items()}
    
        return results
