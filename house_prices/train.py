import pandas as pd
from house_prices.preprocess import select_features
from house_prices.preprocess import get_train_test_sets
from house_prices.preprocess import ordinal_encode_data
from house_prices.preprocess import one_hot_encode_data
from house_prices.preprocess import scale_data
from house_prices.preprocess import train_model
from house_prices.preprocess import ordinal_encode_data_transform
from house_prices.preprocess import one_hot_encode_data_transform
from house_prices.preprocess import scale_data_transform
from house_prices.preprocess import prediction
from house_prices.preprocess import compute_rmsle


def evaluate_model(df: pd.DataFrame) -> dict[str, str]:
    X, y = select_features(df)
    X_train, X_test, y_train, y_test = get_train_test_sets(X, y, 0.25, 0)
    X_train = ordinal_encode_data(X_train, 'SaleCondition')
    X_train = one_hot_encode_data(X_train, 'LotShape')
    X_train = scale_data(X_train, 'OverallQual',
                         'GrLivArea', 'BedroomAbvGr', 'GarageArea')
    train_model(X_train, y_train)
    X_test = ordinal_encode_data_transform(X_test, 'SaleCondition')
    X_test = one_hot_encode_data_transform(X_test, 'LotShape')
    X_test = scale_data_transform(X_test, 'OverallQual',
                                  'GrLivArea', 'BedroomAbvGr', 'GarageArea')
    y_pred = prediction(X_test)
    compute_rmsle(y_test, y_pred)
    return {'rsme': compute_rmsle(y_test, y_pred)}
