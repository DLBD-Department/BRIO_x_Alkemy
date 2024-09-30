import numpy as np
import logging

from datetime import datetime
import os
import psutil


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

        usage = {}
        total_memory, used_memory, free_memory = map(
            int, os.popen('free -t -m').readlines()[-1].split()[1:])

        usage['start'] = {}
        start_dateTime = datetime.now()
        usage['start']['dateTime'] = start_dateTime.strftime("%Y-%m-%d %H:%M:%S")
        usage['start']['total_memory'] = str(total_memory)
        usage['start']['used_memory'] = str(round((used_memory / total_memory) * 100, 2))
        usage['start']['cpu_percent'] = str(psutil.cpu_percent(4))

        #tot number features=conditioning + root (+1)
        n_features_total = len(conditioning_variables) + 1

        hazard_overall = 0
        hazard_overall_max = 0
        # Iterating over each reference distribution, if available (FreqVsRef)
        # In case of FreqVsFreq, there will be a single iteration
        num_iterations = len(self.as_list(overall_result['distance']))
        for k in np.arange(0, num_iterations):

            # test result, threshold, num_samples, boolean, num_used_features
            #TODO use dict instead, and use explicit keys for readibility
            test_results = []
            test_results.append((
                self.as_list(overall_result['distance'])[k],
                overall_result['computed_threshold'],
                tot_observations,
                self.as_list(overall_result['df_vs_thr'])[k],
                1  #for the overall test, only 1 feature used, the root variable
            ))

            for group_name, group in conditioned_results.items():
                if (self.as_list(group[1])[k] is not None):
                    test_results.append(
                        (
                            self.as_list(group[1])[k],  #test result
                            group[3],  #threshold
                            group[0],  #num_samples
                            self.as_list(group[2])[k],  #boolean
                            len(group_name.split("&")) + 1  #num_used_features, cond.+root
                        )
                    )

            if weight_logic == "group":
                #T_i in Risk Function document
                weight_denominator = 0
                for line in test_results:
                    weight_denominator += n_features_total - line[4] + 1
            elif weight_logic == "individual":
                #S_i in Risk Function document
                weight_denominator = np.sum([x[4] for x in test_results])
            else:
                raise Exception('Only "group" or "individual" are allowed for parameter weight_logic')

            hazards = []
            hazard = 0
            for line in test_results:
                if weight_logic == "group":
                    c_info = n_features_total - line[4] + 1
                    weight = c_info / weight_denominator
                elif weight_logic == "individual":
                    weight = line[4] / weight_denominator
                else:
                    raise Exception('Only "group" or "individual" are allowed for parameter weight_logic')

                q = line[2] / tot_observations
                e = line[0] - line[1]
                hazard_cumulative = weight * q * abs(e) ** (1. / 3.) * line[1] ** (1. / 3.)
                delta = 0  #when line[3] == True or 'Not enough observations'
                if line[3] == False:
                    delta = 1
                hazard = delta * hazard_cumulative
                #append single line hazard to hazards array
                hazards.append(hazard)
                hazard_overall += hazard
                hazard_overall_max += hazard_cumulative
        #append hazard_overall to hazards array
        hazards.insert(0, hazard_overall)
        hazards.insert(len(hazards) + 1, hazard_overall_max)

        usage['end'] = {}
        end_dateTime = datetime.now()
        usage['end']['dateTime'] = end_dateTime.strftime("%Y-%m-%d %H:%M:%S")
        total_memory, used_memory, free_memory = map(
            int, os.popen('free -t -m').readlines()[-1].split()[1:])
        usage['end']['total_memory'] = str(total_memory)
        usage['end']['used_memory'] = str(round((used_memory / total_memory) * 100, 2))
        usage['end']['cpu_percent'] = str(psutil.cpu_percent(4))
        usage['timing'] = str((end_dateTime - start_dateTime).total_seconds() / 60)

        response = {
            'hazards': hazards,
            'usage': usage
        }
        #return hazards  # hazards = [individual risk, unconditioned hazard, conditioned hazards, ..., hazard_overall_max]
        return response
