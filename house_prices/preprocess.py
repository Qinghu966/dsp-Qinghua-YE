import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
import joblib


svc_model = joblib.load('../dsp-Qinghua-YE/models/model.joblib',
                        mmap_mode=None)

ordinalEncoder = joblib.load('../dsp-Qinghua-YE/models/ordinal_encoder.joblib',
                             mmap_mode=None)
oneHotEncoder = joblib.load('../dsp-Qinghua-YE/models/one_hot_encoder.joblib',
                            mmap_mode=None)
standardScaler = joblib.load('../dsp-Qinghua-YE/models/scaler.joblib',
                             mmap_mode=None)


def select_features(df):
    X = df[['SalePrice', 'OverallQual', 'GrLivArea',
            'BedroomAbvGr', 'GarageArea', 'LotShape', 'SaleCondition']]
    X = X.dropna().reset_index(drop=True)
    y = df['SalePrice'].values.reshape(-1, 1)
    X = X.drop('SalePrice', axis=1)
    return X, y


def get_train_test_sets(X_feature, y_feature, size, random):
    X_train, X_test, y_train, y_test = train_test_split(
        X_feature, y_feature, test_size=size, random_state=random)
    return X_train, X_test, y_train, y_test


def ordinal_encode_data(df, ord_encode_column):
    df[ord_encode_column] = ordinalEncoder.fit_transform(
        df[[ord_encode_column]])
    return df


def one_hot_encode_data(df, one_hot_encode_column):
    encoded_columns = oneHotEncoder.fit_transform(
        df[one_hot_encode_column].values.reshape(-1, 1))
    encoded_columns_names = oneHotEncoder.get_feature_names_out(
        df[[one_hot_encode_column]].columns)
    encoded_df = pd.DataFrame(data=encoded_columns,
                              columns=encoded_columns_names,
                              index=df[[one_hot_encode_column]].index)
    df = df.copy().join(encoded_df)
    df = df.drop(one_hot_encode_column, axis=1)
    return df


def scale_data(df, scale_column1, scale_column2, scale_column3, scale_column4):
    df[[scale_column1, scale_column2,
        scale_column3, scale_column4]] = standardScaler.fit_transform(
        df[[scale_column1, scale_column2, scale_column3, scale_column4]])
    return df


def prediction(predict_data):
    y_pred = svc_model.predict(predict_data)
    return y_pred


def compute_rmsle(y_test: np.ndarray,
                  y_pred: np.ndarray, precision: int = 2) -> float:
    rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
    return round(rmsle, precision)
