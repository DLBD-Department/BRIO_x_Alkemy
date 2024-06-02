import unittest
from pickle import load
from sklearn.model_selection import train_test_split

from brio.utils.Preprocessing import Preprocessing
from brio.risk.RiskCalculator import RiskCalculator
from brio.risk.HazardFromBiasDetectionCalculator import HazardFromBiasDetectionCalculator

class TestRiskCalculator(unittest.TestCase):
    
    def setUp(self):
        input_data_path = "./tests/unit/test_data/data.csv"
        pp = Preprocessing(input_data_path, "default")
        X, Y = pp.read_dataframe()
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=420)
        self.n_obs = X_test.shape[0]
        self.conditioning_variables = ['x3_education', 'x4_marriage']

        #### Import FreqVsRef results ####
        with open("./tests/unit/test_data/results_test_compare_root_variable_conditioned_groups_with_KL_and_ref_distribution.pkl", "rb") as file:
            self.results_ref_conditioned = load(file)

        with open("./tests/unit/test_data/results_test_compare_root_variable_groups_with_KL_and_ref_distribution.pkl", "rb") as file:
            self.results_ref_overall = load(file)

        #### Import FreqVsFreq results ####
        with open("./tests/unit/test_data/results_test_compare_root_variable_groups_with_TVD.pkl", "rb") as file:
            self.results_freq_overall = load(file)

        with open("./tests/unit/test_data/results_test_compare_root_variable_conditioned_groups_with_TVD.pkl", "rb") as file:
            self.results_freq_conditioned = load(file)


    def test_risk_calculator(self):
        '''
        Test the risk_calculator method, using a set of FreqVsFreq and
        FreqVsRef results. The overall risk measure is a combination 
        coming from all the input results. 
        '''
        hc = HazardFromBiasDetectionCalculator()

        hazard_ref = hc.compute_hazard_from_freqvsfreq_or_freqvsref(self.results_ref_overall, 
                        self.results_ref_conditioned, 
                        self.n_obs,
                        self.conditioning_variables)
        
        hazard_freq = hc.compute_hazard_from_freqvsfreq_or_freqvsref(self.results_freq_overall, 
                        self.results_freq_conditioned, 
                        self.n_obs,
                        self.conditioning_variables)
        
        rc = RiskCalculator()
        risk = rc.compute_risk(test_hazards=[hazard_ref, hazard_freq])

        self.assertAlmostEqual(risk, 0.018531866342260846, delta=1e-8)

if __name__ == '__main__':
    unittest.main()