import numpy as np
from itertools import chain
from itertools import combinations as itertools_combinations


class BiasDetector:

    def powerset(self, iterable):
        '''
        Recipe from itertool documentation. 
        https://docs.python.org/3/library/itertools.html#itertools-recipes
        '''
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        return chain.from_iterable(itertools_combinations(s, r) for r in range(len(s) + 1))

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

        freq_list = []
        abs_freq_list = []
        for label in root_variable_labels:
            dataframe_subset = dataframe.loc[
                dataframe[root_variable] == label
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



