from scipy.stats import entropy
from itertools import combinations
from math import exp

class KLDivergence:

    def __init__(self, 
        aggregating_function=max,
        normalization="D1"):
        '''
        normalization: str, it can be D0, D1 or D2. This is used to force KL
        to have [0, 1] support. 
            D0: no normalization is performed
            D1: D_1 = 1 - 1/(1+D)
            D2: D_2 = 1 - exp(-D)
        '''
        
        # function needed to aggregate distances for multi-class comparisons
        self.aggregating_function = aggregating_function
        if normalization == "D0":
            self.normalization_function = lambda x: x
        elif normalization == "D1":
            self.normalization_function = lambda x: 1 - 1/(1+x)
        elif normalization == "D2":
            self.normalization_function = lambda x: 1 - exp(-x)
        else:
            raise Exception("Only D0, D1 and D2 are supported as normalization methods.")


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

        divergences = []
        for ref, obs in zip(reference_distribution, observed_distribution):
            kl = entropy(pk=ref, qk=obs)
            divergence = self.normalization_function(kl)
            divergences.append(divergence)

        return divergences

    
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

        divergences = []
        for pair in combinations(observed_distribution, 2):
            kl = ( entropy(pk=pair[0], qk=pair[1]) + entropy(pk=pair[1], qk=pair[0]) )/2
            divergence = self.normalization_function(kl)
            divergences.append(divergence)
        
        divergence = self.aggregating_function(divergences)

        return divergence
