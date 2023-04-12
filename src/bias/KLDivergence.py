from scipy.stats import entropy

class KLDivergence:

    def __init__(self):
        pass

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

        divergences = [
                entropy(pk=ref, qk=obs) for ref, obs in zip(
                    reference_distribution, observed_distribution
                    )
                ]

        return divergences

    
    def compute_distance_between_frequencies(self, 
            observed_distribution):
        '''
        observed_distribution: list of numpy arrays, 
            each of them containing the distribution target_variable | root_variable. 
            e.g. [ array(female_0, female_1), array(male_0, male_1) ]  
            The lenght of the list is given by the number of categories of the root variable.
            The shape of each array is given by the number of labels of target_variable.
            
        It works for any number of labels of the target variable, but it is 
        currently implemented only for the binary root_variable case.
        '''


        # TODO generalize to root_variable with more than 2 classes. 
        # At the moment, this compare the freqs of class 1 vs class 2. 
        divergence = (
            entropy(pk=observed_distribution[0], qk=observed_distribution[1]) + entropy(pk=observed_distribution[1], qk=observed_distribution[0])
            )/2


        return divergence
