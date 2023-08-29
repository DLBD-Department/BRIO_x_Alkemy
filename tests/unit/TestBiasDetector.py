import unittest
import sys
from pickle import load
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from src.utils.Preprocessing import Preprocessing
from src.bias.FreqVsFreqBiasDetector import FreqVsFreqBiasDetector
from src.bias.FreqVsRefBiasDetector import FreqVsRefBiasDetector

import logging

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

        self.df_with_predictions = X_test.reset_index(drop=True).assign(predictions=predicted_values, predicted_probs=predicted_prob[:,1])

        #### Reference Distribution ####
        # to use when target is a binary class
        male_0_ref = 75/100
        male_1_ref = 25/100

        female_0_ref = 75/100
        female_1_ref = 25/100

        self.ref = [np.array([male_0_ref, male_1_ref]), np.array([female_0_ref, female_1_ref])]

        #to use when target is a probability (and we split in 10 bins)
        male_ref_prob = [0.2,0.15,0.15,0.1,0.1,0.1,0.05,0.05,0.05,0.05]
        female_ref_prob = [0.2,0.15,0.15,0.1,0.1,0.1,0.05,0.05,0.05,0.05]
        self.ref_prob = [np.array(male_ref_prob), np.array(female_ref_prob)]


    def test_compare_root_variable_groups_with_TVD(self):
        '''
        Test the compare_root_variable_groups method against a 
        known data set and model when the target variable is a class.
        It uses the Total Variation Distance and max as aggregating function.
        '''
        bd = FreqVsFreqBiasDetector(distance="TVD", aggregating_function=max, A1="high")

        results = bd.compare_root_variable_groups(
                    dataframe=self.df_with_predictions,
                    target_variable='predictions',
                    root_variable='x2_sex',
                    threshold=0.1)

        self.assertEqual(results[0], 0.025269625352224545)

    
    def test_compare_root_variable_groups_with_JS(self):
        '''
        Test the compare_root_variable_groups method against a 
        known data set and model when the target variable is a class.
        It uses the JS Divergence and max as 
        aggregating function. 
        '''
        bd = FreqVsFreqBiasDetector(distance="JS", aggregating_function=max, A1="high")

        results = bd.compare_root_variable_groups(
                    dataframe=self.df_with_predictions,
                    target_variable='predictions',
                    root_variable='x2_sex',
                    threshold=0.1)

        self.assertEqual(results[0], 0.0011441803173238346)


    def test_compare_root_variable_groups_probs(self):
        '''
        Test the compare_root_variable_groups method against a 
        known data set and model when the target variable is a probability
        '''
        for dist, result in zip(["TVD","JS"], [0.05588711, 0.005517027]):
            bd = FreqVsFreqBiasDetector(distance=dist, aggregating_function=max, A1="high", target_variable_type='probability')

            results = bd.compare_root_variable_groups(
                        dataframe=self.df_with_predictions,
                        target_variable='predicted_probs',
                        root_variable='x2_sex',
                        threshold=0.1,
                        n_bins=10)
            with self.subTest():
                self.assertAlmostEqual(results[0], result, delta=1e-8)


    def test_compare_root_variable_groups_with_KL_and_ref_distribution(self):
        '''
        Test the compare_root_variable_groups method against a 
        known data set and model when the target variable is a class.
        It uses the symmetrical KL divergence and max as aggregating function(default).
        It uses a reference distribution for the bias detection.
        '''
        bd = FreqVsRefBiasDetector(normalization="D1", A1="high")

        results = bd.compare_root_variable_groups(
                    dataframe=self.df_with_predictions,
                    target_variable='predictions',
                    root_variable='x2_sex',
                    threshold=0.1,
                    reference_distribution=self.ref)
        self.assertEqual(results[0], [0.07485260878313427, 0.11543085607355452])


    def test_compare_root_variable_groups_with_KL_and_ref_distribution_probs(self):
        '''
        Test the compare_root_variable_groups method against a 
        known data set and model when the target variable is a probability.
        It uses the symmetrical KL divergence and max as aggregating function (default).
        It uses a reference distribution for the bias detection.
        '''
        bd = FreqVsRefBiasDetector(normalization="D1", A1="high", target_variable_type='probability')

        results = bd.compare_root_variable_groups(
                    dataframe=self.df_with_predictions,
                    target_variable='predicted_probs',
                    root_variable='x2_sex',
                    threshold=0.1,
                    reference_distribution=self.ref_prob,
                    n_bins=10)
        with self.subTest():
            self.assertAlmostEqual(results[0][0], 0.332536911, delta=1e-8)  
        with self.subTest():
            self.assertAlmostEqual(results[0][1], 0.395066879, delta=1e-8)


    def test_compare_root_variable_conditioned_groups_with_TVD(self):
        '''
        Test the compare_root_variable_conditioned_groups method against
        a known dataset and a model when the target variable is a class.
        It uses the Total Variation Distance and max as 
        aggregating function.
        '''

        bd = FreqVsFreqBiasDetector(distance="TVD", aggregating_function=max, A1="high")

        results = bd.compare_root_variable_conditioned_groups(
            self.df_with_predictions,
            'predictions',
            'x2_sex',
            ['x3_education', 'x4_marriage'],
            0.1,
            min_obs_per_group=30)

        violations = {k: v for k, v in results.items() if not v[2]}

        self.assertEqual(len(violations), 0)


    def test_compare_root_variable_conditioned_groups_with_JS(self):
        '''
        Test the compare_root_variable_conditioned_groups method against
        a known dataset and a model when the target variable is a class.
        It uses the symmetrical KL Divergence and max as 
        aggregating function.
        '''

        bd = FreqVsFreqBiasDetector(distance="JS", aggregating_function=max, A1="high")

        results = bd.compare_root_variable_conditioned_groups(
            self.df_with_predictions,
            'predictions',
            'x2_sex',
            ['x3_education', 'x4_marriage'],
            0.1,
            min_obs_per_group=30)

        violations = {k: v for k, v in results.items() if not v[2]}

        self.assertEqual(len(violations), 0)


    def test_compare_root_variable_conditioned_groups_probs(self):
        '''
        Test the compare_root_variable_conditioned_groups method against
        a known dataset and a model when the target variable is a probability.
        It uses max as aggregating function.
        '''


        for dist, result in zip(["TVD","JS"], [4,2]):
            bd = FreqVsFreqBiasDetector(distance=dist, aggregating_function=max, A1="high", target_variable_type='probability')

            results = bd.compare_root_variable_conditioned_groups(
                dataframe=self.df_with_predictions,
                target_variable='predicted_probs',
                root_variable='x2_sex',
                conditioning_variables=['x3_education', 'x4_marriage'],
                threshold=0.1,
                min_obs_per_group=30,
                n_bins=10)

            violations = {k: v for k, v in results.items() if not v[2]}
            with self.subTest():
                self.assertEqual(len(violations), result)


    def test_compare_root_variable_conditioned_groups_with_KL_and_ref_distribution(self):
        '''
        Test the compare_root_variable_conditioned_groups method against
        a known dataset and a model when the target variable is a class.
        It uses the symmetrical KL Divergence and max as 
        aggregating function.
        It uses a reference distribution for the bias detection. 
        '''

        bd = FreqVsRefBiasDetector(normalization="D1", adjust_div=None, A1="high")

        results = bd.compare_root_variable_conditioned_groups(
            self.df_with_predictions,
            'predictions',
            'x2_sex',
            ['x3_education', 'x4_marriage'],
            reference_distribution=self.ref,
            min_obs_per_group=30,
            threshold=0.1)

        violations = {k: v for k, v in results.items() if (not v[2][0] or not v[2][1])}
        self.assertEqual(len(violations), 9)
        

    def test_compare_root_variable_conditioned_groups_with_KL_and_ref_distribution_probs(self):
        '''
        Test the compare_root_variable_conditioned_groups method against
        a known dataset and a model when the target variable is a probability.
        It uses the symmetrical KL Divergence and max as 
        aggregating function.
        It uses a reference distribution for the bias detection. 
        '''

        for adjust_div,result in zip([None, 'zero'], [16,12]):
            bd = FreqVsRefBiasDetector(normalization="D1", adjust_div=adjust_div, A1="high", target_variable_type='probability')

            results = bd.compare_root_variable_conditioned_groups(
                dataframe=self.df_with_predictions,
                target_variable='predicted_probs',
                root_variable='x2_sex',
                conditioning_variables=['x3_education', 'x4_marriage'],
                reference_distribution=self.ref_prob,
                min_obs_per_group=30,
                threshold=0.1,
                n_bins=10)

            violations = {k: v for k, v in results.items() if (not v[2][0] or not v[2][1])}
            with self.subTest():
                self.assertEqual(len(violations), result)


    def test_compare_freqvsfreq_distances_from_proba_and_distances_from_classes(self):
        '''
        Check that the freqvsfreq distances computed from the predicted probabilities, with two bins, 
        equal those computed using the predicted classes (for a binary classification problem)
        '''
                
        for dist in ["TVD","JS"]:
            bd_class_freq = FreqVsFreqBiasDetector(distance=dist, target_variable_type='class')
            results_class = bd_class_freq.compare_root_variable_groups(
                dataframe=self.df_with_predictions,
                target_variable='predictions',
                root_variable='x2_sex',
                threshold=None
            )
            bd_prob_freq = FreqVsFreqBiasDetector(distance=dist, target_variable_type='probability')
            results_prob = bd_prob_freq.compare_root_variable_groups(
                dataframe=self.df_with_predictions,
                target_variable='predicted_probs',
                root_variable='x2_sex',
                threshold=None,
                n_bins=2
            )
            with self.subTest():
                self.assertEqual(results_class, results_prob)

    def test_compare_freqvsref_distances_from_proba_and_distances_from_classes(self):
        '''
        Check that the freqvsfref distances computed from the predicted probabilities, with two bins, 
        equal those computed using the predicted classes (for a binary classification problem)
        '''
                
        for adjust_div in [None, 'zero']:
            bd_class_freq = FreqVsRefBiasDetector(adjust_div=adjust_div, target_variable_type='class')
            results_class = bd_class_freq.compare_root_variable_groups(
                dataframe=self.df_with_predictions,
                target_variable='predictions',
                root_variable='x2_sex',
                reference_distribution=self.ref,
                threshold=None
            )
            bd_prob_freq = FreqVsRefBiasDetector(adjust_div=adjust_div, target_variable_type='probability')
            results_prob = bd_prob_freq.compare_root_variable_groups(
                dataframe=self.df_with_predictions,
                target_variable='predicted_probs',
                root_variable='x2_sex',
                reference_distribution=self.ref,
                threshold=None,
                n_bins=2
            )

            with self.subTest():
                self.assertEqual(results_class, results_prob)



if __name__ == '__main__':
    unittest.main()

