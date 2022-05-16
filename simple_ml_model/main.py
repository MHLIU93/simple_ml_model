import joblib
import os

from utils.load_data import load_data
from utils.save_output import save_output

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'models/best_model.pkl')

def predict(): 
    # load data 
    df = load_data()
    X = df.iloc[:10,:-1]

    # load best model
    best_model = joblib.load(filename)
    y_hat = best_model.predict(X)
    print('prediction successful!')

    # save data into database
    save_output(y_hat)
    print("prediction output saved.")

if __name__ == '__main__': 
    predict()
