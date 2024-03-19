import numpy as np

class RiskCalculator:

    def compute_risk(self, test_hazards):
        '''
        Computes the overall risk using the hazards coming from the 
        different Bias and Opacity tests. 

        Args:
            test_hazards: list of hazards computed for a set of tests

        Returns:
            risk: num, the overall measure of risk
        '''
        # test_hazards = [list_of_hazards]
            
        risk = np.sum(test_hazards)/len(test_hazards)**2
        
        return risk