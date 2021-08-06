from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from starlette.responses import FileResponse
from starlette.routing import Host 
import uvicorn

app = FastAPI(debug=True)
templates = Jinja2Templates(directory="templates/")
test_samples_idx = [val for val in range(1, 2948)]
default_data = {'id':1, 'X1': [1, 1], 'Y1': [1,2], 'Z1': [1,2], 'X2': [1,2], 'Y2': [1,2], 'Z2': [1,2], }

@app.get("/")
async def read_index(request: Request):
    return templates.TemplateResponse('index.html', context={'request': request, 'options_list': test_samples_idx, 'instance_data':default_data})

@app.post('/get_test_instance')
async def load_instance(request: Request, doc_id: int=0):
    print(doc_id)
    output = {'id':doc_id, 'X1': [doc_id, doc_id], 'Y1': [1,2], 'Z1': [1,2], 'X2': [1,2], 'Y2': [1,2], 'Z2': [1,2], }
    return output

@app.post('/predict_instance')
async def load_instance(request: Request, doc_id: int=Form(...)):
    print(doc_id)
    output = {'id':doc_id, 'X1': [doc_id, doc_id], 'Y1': [1,2], 'Z1': [1,2], 'X2': [1,2], 'Y2': [1,2], 'Z2': [1,2], }
    return templates.TemplateResponse('index.html', context={'request': request, 'options_list': test_samples_idx, 'instance_data':output, 'output': output})

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port='8080')