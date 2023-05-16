from src.bias.TotalVariationDistance import TotalVariationDistance
from src.data_processing.Preprocessing import Preprocessing
from src.bias.BiasDetector import BiasDetector
from sklearn.model_selection import train_test_split
import numpy as np
from pickle import dump, load
import pandas as pd
import statistics

input_data_path = "./data/raw_data/uci-default-of-credit-card/data/data.csv"

with open("notebooks/mlruns/1/1e4a0667c7a64cbe8c7b023410e5781c/artifacts/model/model.pkl", "rb") as file:
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

d = TotalVariationDistance(aggregating_function=statistics.stdev)
bd = BiasDetector(distance=d)

results = bd.compare_root_variable_conditioned_groups(
    df_with_predictions,
    'predictions',
    'x3_education',
    ['x2_sex', 'x4_marriage'],
    0.1)

print(results)
