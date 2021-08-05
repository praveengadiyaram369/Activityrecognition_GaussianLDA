# -*- coding: utf-8 -*-


# 1. Library imports

import uvicorn
import pandas as pd
from fastapi import FastAPI
from Activity import Activity_Data
from utils import glda_clustering

# 2. Create the app object
app = FastAPI()


# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, World'}

@app.post('/predict')
def predict_Activity(data:Activity_Data):
    data=data.dict()
    main_df=pd.DataFrame(data)
    #print(main_df)
    activity = glda_clustering(main_df)

    return {'predicted_label':activity}

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn app:app --reload