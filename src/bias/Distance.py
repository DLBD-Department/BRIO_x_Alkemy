class Distance:

    def __init__(self):
        pass

    def compute_distance_from_reference(self, observed_distribution,
            reference_distribution):
        '''
        observed_distribution: list of numpy arrays
        reference_distribution: list of numpy arrays
        '''

        distances = [
                max(abs(ref - obs)) for ref, obs in zip(
                    reference_distribution, observed_distribution
                    )
                ]

        return distances

    
    def compute_distance_between_frequencies(self, observed_distribution):
        # It works only if the observed distribution is splitted in two groups 
        # only (like male and females)
        distance = max(abs(observed_distribution[0] - observed_distribution[1]))

        return distance
