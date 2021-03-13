from fastapi import FastAPI
from starlette.responses import JSONResponse
from joblib import load
import pandas as pd
from fastapi.encoders import jsonable_encoder
#from pydantic import BaseModel
from torch.utils.data import DataLoader

from model import format_features, apply_scaler, single_tensor, ClassificationEmbdNN, get_model, predict

app = FastAPI()



@app.get("/")
def read_root():
    return {"Displaying a brief description of the project objectives, list of endpoints, expected input parameters and output format of the model, link to the Github repo   related to this project"}


@app.get('/health', status_code=200)
def healthcheck():
    return 'GMM Clustering is all ready to go!'


@app.post("/beer/type/")
async def predict_beer_style(brewery_name: int, review_aroma: int, review_appearance: int, review_palate: int, review_taste: int):
    
    features = format_features(brewery_name,review_aroma, review_appearance, review_palate, review_taste)
    obs = pd.DataFrame(features)
    # scale num cols
    obs = apply_scaler(obs)
    
    #transform to embed object
    obs_tensor = single_tensor(obs)
    
    #predict on embed obj
    model = get_model()
    
    #return predictions as text string
    answer = predict(obs_tensor, model, single=True)
    
    
    return answer

@app.post("/beers/type/")
async def predict_beer_style_multi(brewery_name: int, review_aroma: int, review_appearance: int, review_palate: int, review_taste: int):
    
    features = format_features(brewery_name,review_aroma, review_appearance, review_palate, review_taste)
    obs = pd.DataFrame(features)
    # scale num cols
    obs = apply_scaler(obs)
    
    #transform to embed object
    obs_tensor = single_tensor(obs)
    
    #predict on embed obj
    model = get_model()
    
    #return predictions as text string
    answer = predict(obs_tensor, model)
    
    
    return answer


@app.get("/model/architecture/")
async def print_model():
    
    model = get_model()
    return model

