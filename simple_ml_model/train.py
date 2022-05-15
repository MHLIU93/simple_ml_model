from datetime import datetime
import joblib
import os

from utils.load_data import load_data
from utils.preprocess_data import preprocess_data
from utils.experiment import experiment
from utils.fit_best_model import fit_best_model

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'models/best_model.pkl')

def train_model(): 
    # load data 
    df = load_data()
    # train test split 
    x_train, x_test, y_train, y_test = preprocess_data(df)

    # run experiments 
    best_params = experiment(x_train, x_test, y_train, y_test)

    # fit the best model 
    best_model = fit_best_model(df, best_params)
    # save best model
    now = datetime.now().strftime('%d-%m-%Y_%H:%M:%S')
    joblib.dump(best_model, filename, compress=1)

if __name__ == '__main__': 
    train_model()
