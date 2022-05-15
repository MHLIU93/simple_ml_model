from datetime import datetime
import joblib

from utils.load_data import load_data

def predict(): 
    # load data 
    df = load_data()
    X = df.iloc[:,:-1]

    # load best model
    filename = 'models/best_model.pkl'
    best_model = model = joblib.load(filename)
    y_hat = best_model.predict(X)
    print('prediction successful!')

if __name__ == '__main__': 
    predict()
