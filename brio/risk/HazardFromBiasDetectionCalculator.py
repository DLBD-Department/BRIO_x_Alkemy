import numpy as np

class HazardFromBiasDetectionCalculator:

    '''
    This class manages the calculation of hazards for the 
    tests within the Bias module
    '''

    def as_list(self, x):
        if type(x) is list:
            return x
        else:
            return [x]

    def compute_hazard_from_freqvsfreq_or_freqvsref(
        self,
        overall_result, 
        conditioned_results, 
        tot_observations,
        conditioning_variables,
        weight_logic="group"):

        '''
        Computes the hazard for a FreqVsFreq or a FreqVsRef analysis. 

        Args:
            overall_result: dict with non-conditioned results from FreqVs* analysis
            conditioned_result: dict with conditioned results from FreqVs* analysis #TODO handle when only overall result is available
            tot_observation: num, total number of data points analyzed 
            conditioning_variables: list, conditioning variables used in FreqVs* analysis
            weight_logic: str, it can be either "group" or "individual", it determines how much each single test will weight on the hazard result
        '''
        
        #tot number features=conditioning + root (+1)
        n_features_total = len(conditioning_variables) + 1
        
        hazard_overall = 0
        # Iterating over each reference distribution, if available (FreqVsRef)
        # In case of FreqVsFreq, there will be a single iteration
        num_iterations = len(self.as_list(overall_result[0]))
        for k in np.arange(0, num_iterations):
        
            # test result, threshold, num_samples, boolean, num_used_features
            test_results = []
            test_results.append((
                            self.as_list(overall_result[0])[k], 
                            overall_result[2], 
                            tot_observations, 
                            self.as_list(overall_result[1])[k],
                            1 #for the overall test, only 1 feature used, the root variable
                        ))

            for group_name, group in conditioned_results.items():
                if (group[1] is not None):
                    test_results.append(
                                        (
                                        self.as_list(group[1])[k], #test result
                                        group[3], #threshold
                                        group[0], #num_samples
                                        self.as_list(group[2])[k], #boolean
                                        len(group_name.split("&"))+1 #num_used_features, cond.+root
                                        )
                                    ) 

            if weight_logic=="group":
                #T_i in Risk Function document
                weight_denominator = 0 
                for line in test_results:
                    weight_denominator += n_features_total - line[4] + 1
            elif weight_logic=="individual":
                #S_i in Risk Function document
                weight_denominator = np.sum([x[4] for x in test_results]) 
            else:
                raise Exception('Only "group" or "individual" are allowed for parameter weight_logic')


            hazard = 0
            for line in test_results:
                if weight_logic=="group":
                    c_info = n_features_total - line[4] + 1
                    weight = c_info/weight_denominator
                elif weight_logic=="individual":
                    weight = line[4]/weight_denominator
                else:
                    raise Exception('Only "group" or "individual" are allowed for parameter weight_logic')

                delta = 1 if line[3]==False else 0
                q = line[2]/tot_observations
                e = line[0] - line[1]
                hazard += delta * weight * q * abs(e)**(1./3.) * line[1]**(1./3.)
                
            hazard_overall+= hazard
            
        return hazard_overall