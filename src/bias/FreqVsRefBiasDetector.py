from .BiasDetector import BiasDetector
from .threshold_calculator import threshold_calculator
from sklearn.utils.extmath import cartesian
from math import exp
from scipy.stats import entropy
from scipy.special import rel_entr
import numpy as np
#todo if kl <0 make it None and add a message analogous to 'not enough observations', es 'too many empty bins'
class FreqVsRefBiasDetector(BiasDetector):

    def __init__(self, normalization="D1", adjust_div=None, A1="high", target_variable_type='class'):
        '''
            distance: which distance will be used to compute the bias detection
            A1: sensitivity parameter used to computer the parametric threshold
            target_variable_type: type of the tgt variable. 'class' or 'probability'
            adjust_kl: when a bin of the observed distribution is 0 the relative combination to KL is inf (ref*log(ref/0)), thus 
                the resulting kl is inf (1 when normalized).
                adjust_div=None: this default behaviour is kept.
                adjust_div='zero': inf contributions are forced to zero (not suggested when you have many empty bins)
                adjust_div='laplace': TBD (add 1 obs in the empy bin)

        '''
        self.A1 = A1

        # Normalization method for KL divergence
        if normalization == "D0":
            self.normalization_function = lambda x: x
        elif normalization == "D1":
            self.normalization_function = lambda x: 1 - 1/(1+x)
        elif normalization == "D2":
            self.normalization_function = lambda x: 1 - exp(-x)
        else:
            raise Exception("Only D0, D1 and D2 are supported as normalization methods.")
        
        self.adjust_div = adjust_div
        self.target_variable_type = target_variable_type


    def compute_distance_from_reference(self, 
            observed_distribution,
            reference_distribution):
        '''
        This function computes a normalized version of the KL divergence between reference distribution
        and observed distribution. 

        Args:
            observed_distribution: list of numpy arrays, 
                each of them containing the distribution target_variable | root_variable. 
                e.g. [ array(female_0, female_1), array(male_0, male_1) ]  
                The lenght of the list is given by the number of categories of the root variable.
                The shape of each array is given by the number of labels of target_variable (if self.target_variable_type=='class') 
                or the number of bins (if self.target_variable_type=='probability').
            reference_distribution: list of numpy arrays, 
                each of them containing a reference distribution target_variable | root_variable. 
                e.g. [ array(female_ref_0, female_ref_1), array(male_ref_0, male_ref_1) ]  
                The lenght of the list is given by the number of categories of the root variable.
                The shape of each array is given by the number of labels of target_variable (if self.target_variable_type=='class') 
                or the number of bins (if self.target_variable_type=='probability').

        Returns:
            divergences: list of n divergences, where n is the number of classes of the root variable
        '''

        divergences = []
        for ref, obs in zip(reference_distribution, observed_distribution):
            if self.adjust_div == None:
                kl = entropy(pk=ref, qk=obs, base=2)
            elif self.adjust_div == 'zero':
                kl_elementwise = rel_entr(ref, obs)
                #convert from base e to base 2
                kl_elementwise /= np.log(2)
                #exlude bins where kl is not defined (obs[i]=0 --> kl[i]=np.inf)
                num_empy_bins = len(kl_elementwise[kl_elementwise==np.inf])
                if num_empy_bins>0:
                    print('Warning:', num_empy_bins, 'out of', len(kl_elementwise), 'bins of the observed distribution are empty. \
    Their relative contribution to KL was forced to 0')
                    kl_elementwise[kl_elementwise==np.inf]=0
                kl = kl_elementwise.sum()
            else:
                raise Exception("Only 'no' and 'zero' are supported as divergence adjustment methods.")
            divergence = self.normalization_function(kl)
            divergences.append(divergence)

        return divergences


    def compare_root_variable_groups(self,
            dataframe,
            target_variable,
            root_variable,
            reference_distribution,
            threshold=None,
            n_bins=10):
        '''
        This function computes the distances between a set of n reference distributions, and the 
        distributions of each n class of root_variable w.r.t. target_variable and check if 
        they are below the given threshold.

        Args:
            dataframe: Pandas DataFrame with features and 
                predictions
            target_variable: variable with the predictions (either prediction lables or probabilities, see self.target_variable_type)
            root_variable: ...
            threshold: value from 0 to 1 used to check the computed distance with. If None, the tool will compute a parametric threshold. 
            n_bins: number of bins used to split the predicted probability (only applies when self.target_variable_type='probability')

        Returns:
            A tuple (
                    [distance_a, distance_b, ..., distance_n], 
                    [distance_a<=computed_threshold, ..., distance_n<=computed_threshold], 
                    computed_threshold
                    )
            where n is the number of classes of root_variable
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
        distance = self.compute_distance_from_reference(freqs, reference_distribution)
        return (distance, [d<=computed_threshold for d in distance], computed_threshold)


    def compare_root_variable_conditioned_groups(self, 
            dataframe,
            target_variable,
            root_variable,
            conditioning_variables,
            reference_distribution,
            min_obs_per_group=30,
            threshold=None,
            n_bins=10):
        '''
        This function computes the distances between a set of n reference distributions, and the 
        distributions of each n class of root_variable w.r.t. target_variable, for each 
        subgroup given by the conditioning variables groups. 

        Args:
            dataframe: Pandas DataFrame with features and 
                predicted labels
            target_variable: variable with the predicted labels
            root_variable: ...
            conditioning_variables: list of strings names of the variables that we use 
                to create groups within the population. 
                Starting from the first variable, a tree of conditions is created;
                for each of the resulting group, we check if the n reference distributions are significantly 
                different from the n predicted (observed) distributions (n is the number of root_variable classes).
            threshold: value from 0 to 1 used to check the computed distance with. If None, the tool will compute a parametric threshold.
            min_obs_per_group: the minimum number of observations needed for the distance computation
            reference_distribution: numpy array of probabilities w.r.t. target_variable (e.g. predictions) for the 
                n groups given by root_variable.
            n_bins: number of bins used to split the predicted probability (only applies when self.target_variable_type='probability')

        Returns:
            A dictionary {group_condition: (
                                            numb_obs_of_group, 
                                            [distance_a, distance_b, ..., distance_n], 
                                            [distance_a>=computed_threshold, distance_b>=computed_threshold, ..., distance_n>=computed_threshold]
                                            computed_threshold
                                            )
                            },
            where distance_a, distance_b, ..., distance_n  are the distances computed for the two categories of root_variable. 
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

                else:
                    conditioned_frequencies[condition] = (num_of_obs, None)
        
        distances = {
                group: (
                    (
                        obs_and_freqs[0], 
                        self.compute_distance_from_reference(obs_and_freqs[1], reference_distribution)
                    ) if obs_and_freqs[1] is not None else (obs_and_freqs[0], None)
                ) for group, obs_and_freqs in conditioned_frequencies.items()
            }

        results = {group: (
                (
                    obs_and_dist[0], 
                    obs_and_dist[1], 
                    [d<=threshold_calculator(A1=self.A1, A2=A2, A3=obs_and_dist[0], default_threshold=threshold) for d in obs_and_dist[1]],
                    threshold_calculator(A1=self.A1, A2=A2, A3=obs_and_dist[0], default_threshold=threshold)
                ) if obs_and_dist[1] is not None else (obs_and_dist[0], obs_and_dist[1], 'Not enough observations')
            ) for group, obs_and_dist in distances.items()}
    
        return results