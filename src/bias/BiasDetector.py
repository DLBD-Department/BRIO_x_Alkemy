from .Distance import Distance
import numpy as np

class BiasDetector:

    def __init__(self):
        self.dis = Distance() 

    def get_frequencies_list(self,
            dataframe,
            labels_variable,
            binary_variable,
            binary_variable_labels):

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


    def compare_binary_variable_conditioned_groups(self, 
            dataframe,
            labels_variable,
            binary_variable,
            conditioning_variable,
            threshold):
        '''
        This functions compares the distance between the two 
        categories of binary_variable as observed in the two
        partitions of dataframe, partitions provided by the
        labels_variable, conditioning it
        to each category of conditioning_variable. The distance 
        is computed and compared to the given threshold.

        Args:
            dataframe: Pandas DataFrame with features and 
                predicted labels
            labels_variable: variable with the predicted labels
            binary_variable: variable that we use to compare the predicted
                labels in two different groups of observation
            conditioning_variable: variable that we use to further split 
                the dataframe. Given each category of conditioning_variable,
                we check if the predicted labels frequencies are significantly 
                different in the two groups of binary_variable.
            threshold: value from 0 to 1 used to check the computed distance with.
        '''

        binary_variable_labels = dataframe[binary_variable].unique()
        
        conditioned_frequencies = {}

        #TODO come implemento il caso in cui voglio esplorare conditioning a più variabili?
        # es. m vs f | education & marriage
        # L'idea che avevo avuto era qualche tipo di ricorsione, ma forse basta fare:
        # itero su tutte le variabili di conditioning e aggiungo dei subset successivi. 
        # Da capire come fare
        if isinstance(conditioning_variable, list):
            for variable in conditioning_variable:
                pass
        else:
            for x in dataframe[conditioning_variable].unique():
                dataframe_subset = dataframe.loc[dataframe[conditioning_variable]==x]
    
                conditioned_frequencies[x] = self.get_frequencies_list(
                        dataframe_subset,
                        labels_variable,
                        binary_variable,
                        binary_variable_labels)
    
#            conditioned_freq = np.array(
#                dataframe[dataframe[conditioning_variable]==x][binary_variable].value_counts(normalize=True)
#                                                .reindex(dataframe[binary_variable].unique(), fill_value=0)
#            )
#
#            conditioned_frequencies[x] = [ 
#                    np.array([conditioned_freq[0], 1-conditioned_freq[0]]), 
#                        np.array([conditioned_freq[1], 1-conditioned_freq[1]]) 
#                    ]
#
        distances = {k: self.dis.compute_distance_between_frequencies(v) for k, v in conditioned_frequencies.items()}
        results = {k: (v, v<=threshold) for k, v in distances.items()}

        return results



        
