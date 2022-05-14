import pandas as pd
from sklearn import model_selection

import utils.ml_pipeline_config as config

test_size = config.params["test_split_ratio"]

def preprocess_data(df):

    x_train, x_test, y_train, y_test = model_selection.train_test_split(df.iloc[:,:-1], 
                                                                        df['label'], 
                                                                        test_size=test_size)
    x_train.name = 'x_train'
    x_test.name = 'x_test'
    y_train.name = 'y_train'
    y_test.name = 'y_test'
    return x_train, x_test, y_train, y_test
