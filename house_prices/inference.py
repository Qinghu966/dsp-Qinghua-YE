import pandas as pd
import numpy as np
from house_prices.preprocess import choose_features
from house_prices.preprocess import ordinal_encode_data_transform
from house_prices.preprocess import one_hot_encode_data_transform
from house_prices.preprocess import scale_data_transform
from house_prices.preprocess import prediction


def model_inference(data: pd.DataFrame) -> np.ndarray:
    X = choose_features(data)
    X = ordinal_encode_data_transform(X, 'SaleCondition')
    X = one_hot_encode_data_transform(X, 'LotShape')
    X = scale_data_transform(X, 'OverallQual',
                             'GrLivArea', 'BedroomAbvGr', 'GarageArea')
    y_pred_infer = prediction(X)
    return y_pred_infer
