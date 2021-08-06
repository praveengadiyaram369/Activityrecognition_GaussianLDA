# -*- coding: utf-8 -*-


# 1. Library imports

import uvicorn
import pandas as pd
from fastapi import FastAPI, Request, Form
from Activity import Activity_Data
from fastapi.templating import Jinja2Templates
from utils import glda_clustering
import os
import numpy as np

# 2. Create the app object
app = FastAPI(debug=True)
templates = Jinja2Templates(directory="templates/")

channel_labels = ['X1', 'Y1', 'Z1', 'X2', 'Y2', 'Z2'] 
test_samples_idx = [val for val in range(1, 2948)]
activity_names_map = {1:'WALKING', 2:'WALKING_UPSTAIRS', 3: 'WALKING_DOWNSTAIRS', 4:'SITTING', 5:'STANDING', 6:'LAYING'}

test_instances = dict()
test_instance_labels = dict()
default_data = dict()

def transform_data(data):

    data_dict = {}
    for idx in range(len(channel_labels)):
        data_dict[channel_labels[idx]] = list(data[idx, :])

    return data_dict
        

def set_default_data():
    global default_data
    default_data = test_instances[1]


def load_test_instances():
    global test_instances
    global test_instance_labels

    test_filepath = os.getcwd() + '/../../data/output_csv/processed_data_test.csv'
    test_data = np.loadtxt(test_filepath, delimiter=',')

    doc_ids = (test_data[:, 0].reshape(2947,128))[:, 0]
    doc_labels = (test_data[:, 1].reshape(2947,128))[:, 0]
    sensory_data = test_data[:, 2:].reshape(2947,6,128)

    for doc_id, doc_label, data in zip(doc_ids, doc_labels, sensory_data):

        doc_id = int(doc_id)
        test_instance_labels[doc_id] = int(doc_label)
        test_instances[doc_id] = transform_data(data)
    
    set_default_data()


# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get("/")
async def read_index(request: Request):
    load_test_instances()
    return templates.TemplateResponse('index.html', context={'request': request, 'options_list': test_samples_idx, 'instance_data':default_data})

@app.post('/get_test_instance')
async def load_instance(request: Request, doc_id: int=Form(...)):
    
    output = test_instances[doc_id]
    return output

@app.post('/predict_instance')
async def predict_Activity(request: Request, doc_id: int=Form(...)):
    
    data = test_instances[doc_id]

    main_df=pd.DataFrame(data)
    activity = glda_clustering(main_df)

    return templates.TemplateResponse('index.html', context={'request': request, 'options_list': test_samples_idx, 'instance_data':default_data, 'prev_instance_data':doc_id, 'output':activity_names_map[activity], 'ground_truth':activity_names_map[test_instance_labels[doc_id]]})

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