import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class Preprocessing:

    def __init__(self, input_data_path, target_classification):
        self.input_data_path = input_data_path
        self.target = target_classification


    def read_dataframe(self):
        df = pd.read_csv(self.input_data_path)
        df[self.target] = df['y_default_payment_next_month']

        Y = df[self.target]
        Y = Y.apply(lambda y: 1 if y is True else 0) 
        X = df.drop(columns=[self.target], axis=1)

        return X, Y


    def preprocess_for_classification(self,
            df,
            fit_ohe=False,
            fitted_ohe=None,
            perform_scaling=False,
            fitted_scaler=None):

        categorical_cols = [
                'x2_sex',
                'x3_education',
                'x4_marriage',
                'x6_pay_0',
                'x7_pay_2',
                'x8_pay_3',
                'x9_pay_4',
                'x10_pay_5',
                'x11_pay_6']

        numerical_cols = [
                'x1_limit_bal',
                'x5_age',
                'x12_bill_amt1', 
                'x13_bill_amt2', 
                'x14_bill_amt3',
                'x15_bill_amt4',
                'x16_bill_amt5', 
                'x17_bill_amt6', 
                'x18_pay_amt1',
                'x19_pay_amt2', 
                'x20_pay_amt3', 
                'x21_pay_amt4', 
                'x22_pay_amt5',
                'x23_pay_amt6']

        numeric_means = df[numerical_cols].mean()
        categ_modes = df[categorical_cols].mode().iloc[0]

        df = df.fillna(numeric_means).fillna(categ_modes)
        X = df

        if perform_scaling:
            if fitted_scaler is None:
                fitted_scaler = StandardScaler().fit(df[numerical_cols])
            scaled = fitted_scaler.transform(df[numerical_cols]) 
            scaled_df = pd.DataFrame(scaled, columns=fitted_scaler.get_feature_names_out(input_features=numerical_cols), index=df.index)
            df = pd.concat([df[categorical_cols], scaled_df], axis=1) 
            X = df

        if fit_ohe:
            if fitted_ohe is None:
                fitted_ohe = OneHotEncoder(
                            handle_unknown='ignore',
                            sparse=False).fit(df[categorical_cols])

            cat_ohe = fitted_ohe.transform(df[categorical_cols])
            ohe_df = pd.DataFrame(cat_ohe, columns=fitted_ohe.get_feature_names_out(input_features = categorical_cols), index = df.index)
            X = pd.concat([df[numerical_cols], ohe_df], axis=1)
               

        return X, fitted_ohe, fitted_scaler


















































