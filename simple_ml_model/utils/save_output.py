import pandas as pd
from sqlalchemy import create_engine

import utils.ml_pipeline_config as config

db_engine = config.params["db_engine"]
db_schema = config.params["db_schema"]
output_table = config.params["db_output_table"] 

def save_output(ids, y_hat):
    df = pd.DataFrame({
        "patient_id": ids, 
        "y_hat": y_hat
    })
    engine = create_engine(db_engine)
    df.to_sql(output_table, engine, schema=db_schema, if_exists='replace', index=False)
