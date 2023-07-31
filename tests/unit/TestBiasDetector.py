import unittest
import sys
from pickle import load
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from src.utils.Preprocessing import Preprocessing
from src.bias.BiasDetector import BiasDetector
from src.bias.TotalVariationDistance import TotalVariationDistance
from src.bias.KLDivergence import KLDivergence
from src.bias.JSDivergence import JSDivergence

class TestBiasDetector(unittest.TestCase):

    def setUp(self):

        #### Model and predictions ####
        input_data_path = "./tests/unit/test_data/data.csv"

        with open("./tests/unit/test_data/model.pkl", "rb") as file:
            classifier = load(file)

        pp = Preprocessing(input_data_path, "default")
        X, Y = pp.read_dataframe()
        X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=420)

        shapes_pre = (X_train.shape[0], X_test.shape[0])
        X_train_ohe, ohe, scaler = pp.preprocess_for_classification(df=X_train, fit_ohe=True, perform_scaling=True)
        X_test_ohe, _, _ = pp.preprocess_for_classification(df=X_test,
                                                            fit_ohe=True,
                                                            fitted_ohe=ohe,
                                                            perform_scaling=True,
                                                            fitted_scaler=scaler)

        predicted_prob = classifier.predict_proba(X_test_ohe)
        predicted_values = classifier.predict(X_test_ohe)

        self.df_with_predictions = pd.concat(
            [X_test.reset_index(drop=True), pd.Series(predicted_values)], axis=1).rename(columns={0:"predictions"})

        #### Reference Distribution ####
        male_0_ref = 75/100
        male_1_ref = 25/100

        female_0_ref = 75/100
        female_1_ref = 25/100

        self.ref = [np.array([male_0_ref, male_1_ref]), np.array([female_0_ref, female_1_ref])]


    def test_compare_root_variable_groups_with_TVD(self):
        '''
        Test the compare_root_variable_groups method against a 
        known data set and model.
        It uses the Total Variation Distance and max as aggregating function.
        '''
        d = TotalVariationDistance(aggregating_function=max)
        bd = BiasDetector(distance=d)

        results = bd.compare_root_variable_groups(
                    dataframe=self.df_with_predictions,
                    target_variable='predictions',
                    root_variable='x2_sex',
                    threshold=0.1)

        self.assertEqual(results[0], 0.025269625352224545)


    def test_compare_root_variable_groups_with_KL(self):
        '''
        Test the compare_root_variable_groups method against a 
        known data set and model.
        It uses the symmetrical KL Divergence and max as 
        aggregating function. 
        '''
        d = KLDivergence(aggregating_function=max)
        bd = BiasDetector(distance=d)

        results = bd.compare_root_variable_groups(
                    dataframe=self.df_with_predictions,
                    target_variable='predictions',
                    root_variable='x2_sex',
                    threshold=0.1)

        self.assertEqual(results[0], 0.0045642159318142195)

    
    def test_compare_root_variable_groups_with_JS(self):
        '''
        Test the compare_root_variable_groups method against a 
        known data set and model.
        It uses the JS Divergence and max as 
        aggregating function. 
        '''
        d = JSDivergence(aggregating_function=max)
        bd = BiasDetector(distance=d)

        results = bd.compare_root_variable_groups(
                    dataframe=self.df_with_predictions,
                    target_variable='predictions',
                    root_variable='x2_sex',
                    threshold=0.1)

        self.assertEqual(results[0], 0.0011441803173238346)


    def test_compare_root_variable_groups_with_TVD_and_ref_distribution(self):
        '''
        Test the compare_root_variable_groups method against a 
        known data set and model.
        It uses the Total Variation Distance and max as aggregating function.
        It uses a reference distribution for the bias detection.
        '''
        d = TotalVariationDistance(aggregating_function=max)
        bd = BiasDetector(distance=d)

        results = bd.compare_root_variable_groups(
                    dataframe=self.df_with_predictions,
                    target_variable='predictions',
                    root_variable='x2_sex',
                    threshold=0.1,
                    reference_distribution=self.ref)

        self.assertEqual(results[0], [0.14886196769456683, 0.12359234234234234])


    def test_compare_root_variable_groups_with_KL_and_ref_distribution(self):
        '''
        Test the compare_root_variable_groups method against a 
        known data set and model.
        It uses the symmetrical KL divergence and max as aggregating function.
        It uses a reference distribution for the bias detection.
        '''
        d = KLDivergence(aggregating_function=max)
        bd = BiasDetector(distance=d)

        results = bd.compare_root_variable_groups(
                    dataframe=self.df_with_predictions,
                    target_variable='predictions',
                    root_variable='x2_sex',
                    threshold=0.1,
                    reference_distribution=self.ref)

        self.assertEqual(results[0], [0.11543085607355452, 0.07485260878313427])

    
    def test_compare_root_variable_groups_with_JS_and_ref_distribution(self):
        '''
        Test the compare_root_variable_groups method against a 
        known data set and model.
        It uses the symmetrical KL divergence and max as aggregating function.
        It uses a reference distribution for the bias detection.
        '''
        d = JSDivergence(aggregating_function=max)
        bd = BiasDetector(distance=d)

        results = bd.compare_root_variable_groups(
                    dataframe=self.df_with_predictions,
                    target_variable='predictions',
                    root_variable='x2_sex',
                    threshold=0.1,
                    reference_distribution=self.ref)

        self.assertEqual(results[0], [0.028351705371207105, 0.01830818274294786])


    def test_compare_root_variable_conditioned_groups_with_TVD(self):
        '''
        Test the compare_root_variable_conditioned_groups method against
        a known dataset and a model.
        It uses the Total Variation Distance and max as 
        aggregating function.
        '''

        d = TotalVariationDistance(aggregating_function=max)
        bd = BiasDetector(distance=d)

        results = bd.compare_root_variable_conditioned_groups(
            self.df_with_predictions,
            'predictions',
            'x2_sex',
            ['x3_education', 'x4_marriage'],
            0.1,
            min_obs_per_group=30)

        violations = {k: v for k, v in results.items() if not v[2]}

        self.assertEqual(len(violations), 0)


    def test_compare_root_variable_conditioned_groups_with_KL(self):
        '''
        Test the compare_root_variable_conditioned_groups method against
        a known dataset and a model.
        It uses the symmetrical KL Divergence and max as 
        aggregating function.
        '''

        d = KLDivergence(aggregating_function=max)
        bd = BiasDetector(distance=d)

        results = bd.compare_root_variable_conditioned_groups(
            self.df_with_predictions,
            'predictions',
            'x2_sex',
            ['x3_education', 'x4_marriage'],
            0.1,
            min_obs_per_group=30)

        violations = {k: v for k, v in results.items() if not v[2]}

        self.assertEqual(len(violations), 1)


    def test_compare_root_variable_conditioned_groups_with_JS(self):
        '''
        Test the compare_root_variable_conditioned_groups method against
        a known dataset and a model.
        It uses the symmetrical KL Divergence and max as 
        aggregating function.
        '''

        d = JSDivergence(aggregating_function=max)
        bd = BiasDetector(distance=d)

        results = bd.compare_root_variable_conditioned_groups(
            self.df_with_predictions,
            'predictions',
            'x2_sex',
            ['x3_education', 'x4_marriage'],
            0.1,
            min_obs_per_group=30)

        violations = {k: v for k, v in results.items() if not v[2]}

        self.assertEqual(len(violations), 0)


    def test_compare_root_variable_conditioned_groups_with_TVD_and_ref_distribution(self):
        '''
        Test the compare_root_variable_conditioned_groups method against
        a known dataset and a model.
        It uses the Total Variation Distance and max as 
        aggregating function.
        It uses a reference distribution for the bias detection. 
        '''

        d = TotalVariationDistance(aggregating_function=max)
        bd = BiasDetector(distance=d)

        results = bd.compare_root_variable_conditioned_groups(
            self.df_with_predictions,
            'predictions',
            'x2_sex',
            ['x3_education', 'x4_marriage'],
            0.1,
            min_obs_per_group=30,
            reference_distribution=self.ref)

        violations = {k: v for k, v in results.items() if (not v[2][0] or not v[2][1])}

        self.assertEqual(len(violations), 16)


    def test_compare_root_variable_conditioned_groups_with_KL_and_ref_distribution(self):
        '''
        Test the compare_root_variable_conditioned_groups method against
        a known dataset and a model.
        It uses the symmetrical KL Divergence and max as 
        aggregating function.
        It uses a reference distribution for the bias detection. 
        '''

        d = KLDivergence(aggregating_function=max)
        bd = BiasDetector(distance=d)

        results = bd.compare_root_variable_conditioned_groups(
            self.df_with_predictions,
            'predictions',
            'x2_sex',
            ['x3_education', 'x4_marriage'],
            0.1,
            min_obs_per_group=30,
            reference_distribution=self.ref)

        violations = {k: v for k, v in results.items() if (not v[2][0] or not v[2][1])}

        self.assertEqual(len(violations), 9)

    
    def test_compare_root_variable_conditioned_groups_with_JS_and_ref_distribution(self):
        '''
        Test the compare_root_variable_conditioned_groups method against
        a known dataset and a model.
        It uses the symmetrical KL Divergence and max as 
        aggregating function.
        It uses a reference distribution for the bias detection. 
        '''

        d = JSDivergence(aggregating_function=max)
        bd = BiasDetector(distance=d)

        results = bd.compare_root_variable_conditioned_groups(
            self.df_with_predictions,
            'predictions',
            'x2_sex',
            ['x3_education', 'x4_marriage'],
            0.1,
            min_obs_per_group=30,
            reference_distribution=self.ref)

        violations = {k: v for k, v in results.items() if (not v[2][0] or not v[2][1])}

        self.assertEqual(len(violations), 2)


if __name__ == '__main__':
    unittest.main()
