from src.bias.TotalVariationDistance import TotalVariationDistance
from src.bias.KLDivergence import KLDivergence
from src.data_processing.Preprocessing import Preprocessing
from src.bias.BiasDetector import BiasDetector
from sklearn.model_selection import train_test_split
import numpy as np
from pickle import dump, load
import pandas as pd
import statistics

male_0_ref = 75/100
male_1_ref = 25/100

female_0_ref = 75/100
female_1_ref = 25/100
ref = [np.array([male_0_ref, male_1_ref]), np.array([female_0_ref, female_1_ref])]

input_data_path = "./data/raw_data/uci-default-of-credit-card/data/data.csv"
with open("./notebooks/mlruns/1/1e4a0667c7a64cbe8c7b023410e5781c/artifacts/model/model.pkl", "rb") as file:
    classifier = load(file)

pp = Preprocessing(input_data_path, "default")
X, Y = pp.read_dataframe()
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=420)

#preprocessing ohe
shapes_pre = (X_train.shape[0], X_test.shape[0])
X_train_ohe, ohe, scaler = pp.preprocess_for_classification(df=X_train, fit_ohe=True, perform_scaling=True)
X_test_ohe, _, _ = pp.preprocess_for_classification(df=X_test,
                                                    fit_ohe=True,
                                                    fitted_ohe=ohe,
                                                    perform_scaling=True,
                                                    fitted_scaler=scaler)

predicted_prob = classifier.predict_proba(X_test_ohe)
predicted_values = classifier.predict(X_test_ohe)

df_with_predictions = pd.concat(
    [X_test.reset_index(drop=True), pd.Series(predicted_values)], axis=1).rename(columns={0:"predictions"})

d = KLDivergence(aggregating_function=max)
bd = BiasDetector(distance=d)

results_1 = bd.compare_root_variable_groups(
            dataframe=df_with_predictions,
            target_variable='predictions',
            root_variable='x2_sex',
            threshold=0.1,
            reference_distribution=ref)

print(results_1)
print("end_results_1")

results_2 = bd.compare_root_variable_conditioned_groups(
    df_with_predictions,
    'predictions',
    'x2_sex',
    ['x3_education', 'x4_marriage'],
    0.1,
    min_obs_per_group=30,
    reference_distribution=ref)

violations = {k: v for k, v in results_2.items() if (not v[2][0] or not v[2][1])}


print(len(violations))
