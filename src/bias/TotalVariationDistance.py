from itertools import combinations

class TotalVariationDistance:

    def __init__(self, 
        aggregating_function=max):
        
        # function needed to aggregate distances for multi-class comparisons
        self.aggregating_function = aggregating_function


    def compute_distance_from_reference(self, 
            observed_distribution,
            reference_distribution):
        '''
        observed_distribution: list of numpy arrays, 
            each of them containing the distribution target_variable | root_variable. 
            e.g. [ array(female_0, female_1), array(male_0, male_1) ]  
            The lenght of the list is given by the number of categories of the root variable.
            The shape of each array is given by the number of labels of target_variable.
        reference_distribution: list of numpy arrays, 
            each of them containing a reference distribution target_variable | root_variable. 
            e.g. [ array(female_ref_0, female_ref_1), array(male_ref_0, male_ref_1) ]  
            The lenght of the list is given by the number of categories of the root variable.
            The shape of each array is given by the number of labels of target_variable.
        '''

        distances = [
                max(abs(ref - obs)) for ref, obs in zip(
                    reference_distribution, observed_distribution
                    )
                ]

        return distances

    
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
        '''

        # Computing the TVD for each pair of distributions
        distance = self.aggregating_function(
                # TVD
                [max( abs( pair[0]-pair[1] ) ) for pair in combinations(observed_distribution, 2)]
            )

        return distance
