import pandas as pd
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

def fit_best_model(df, best_params): 
    
    pipe = Pipeline([('scaler', StandardScaler()),
                 ('pca', PCA(n_components = best_params['best_pca_components'].values[0])),
                 ('log_reg', LogisticRegression(C=best_params['best_logreg_c'].values[0]))
                 ])     
    pipe.fit(df.iloc[:,:-1], df['label'])
    return pipe 
