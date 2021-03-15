from fastapi import FastAPI, Query
from starlette.responses import JSONResponse
from joblib import load
import pandas as pd
from fastapi.encoders import jsonable_encoder
#from pydantic import BaseModel
from torch.utils.data import DataLoader
import torch
from typing import List, Optional

from model import format_features, format_features_multi, apply_scaler, single_tensor, get_model, predict, EmbeddingDataset

app = FastAPI()

# embedding class - need to load model
class ClassificationEmbdNN(torch.nn.Module):

    def __init__(self, emb_dims, no_of_cont=None):
        super(ClassificationEmbdNN, self).__init__()

        self.emb_layers = torch.nn.ModuleList([torch.nn.Embedding(x, y)
                                               for x, y in emb_dims])

        no_of_embs = sum([y for x, y in emb_dims])
        self.no_of_embs = no_of_embs
        self.emb_dropout = torch.nn.Dropout(0.2)

        self.no_of_cont = 0
        if no_of_cont:
            self.no_of_cont = no_of_cont
            self.bn_cont = torch.nn.BatchNorm1d(no_of_cont)

        self.fc1 = torch.nn.Linear(in_features=self.no_of_embs + self.no_of_cont,
                                   out_features=208)
        self.dropout1 = torch.nn.Dropout(0.2)
        self.bn1 = torch.nn.BatchNorm1d(208)
        self.act1 = torch.nn.ReLU()

        self.fc2 = torch.nn.Linear(in_features=208,
                                   out_features=208)
        self.dropout2 = torch.nn.Dropout(0.2)
        self.bn2 = torch.nn.BatchNorm1d(208)
        self.act2 = torch.nn.ReLU()

        #         self.fc3 = torch.nn.Linear(in_features=256,
        #                                    out_features=64)
        #         self.dropout3 = torch.nn.Dropout(0.2)
        #         self.bn3 = torch.nn.BatchNorm1d(64)
        #         self.act3 = torch.nn.ReLU()

        self.fc3 = torch.nn.Linear(in_features=208,
                                   out_features=104)
        self.act3 = torch.nn.Softmax()

    def forward(self, x_cat, x_cont=None):
        if self.no_of_embs != 0:
            x = [emb_layer(x_cat[:, i])
                 for i, emb_layer in enumerate(self.emb_layers)]

            x = torch.cat(x, 1)
            x = self.emb_dropout(x)

        if self.no_of_cont != 0:
            x_cont = self.bn_cont(x_cont)

            if self.no_of_embs != 0:
                x = torch.cat([x, x_cont], 1)
            else:
                x = x_cont

        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.bn2(x)
        x = self.act2(x)

        #         x = self.fc3(x)
        #         x = self.dropout3(x)
        #         x = self.bn3(x)
        #         x = self.act3(x)

        x = self.fc3(x)
        x = self.act3(x)

        return x

# load model obj
model = ClassificationEmbdNN(emb_dims=[[5742,252]],no_of_cont=4)
# set to trained dict of weights
model.load_state_dict(torch.load('../models/embed_3layers.pt'))

@app.get("/")
def read_root():
    return {"Predicting a style of beer according to ratings of taste, aroma, appearance and palate scores /n "
            "Endpoints:"
            "‘/’ (GET): Displaying a brief description of the project objectives expected input parameters and output format of the model, link to the Github repo related to this project"
            "‘/health/’ (GET): Returning status code 200 with a string with a welcome message of your choice"
            "‘/beer/type/’ (POST): Returning prediction for a single input only"
            "‘/beers/type/’ (POST): Returning predictions for a multiple inputs"
            "‘/model/architecture/’ (GET): Displaying the architecture of your Neural Networks (listing of all layers with their types)"
            "Inputs for brewery_name should be an integer between 1-5742, "
            "link to the Github repos:  https://github.com/Reasmey/beer_api"
            "https://github.com/Reasmey/adsi_beer_app"}



@app.get('/health', status_code=200)
def healthcheck():
    return 'Glug glug glug'


@app.post("/beer/type/")
async def predict_beer_style(brewery_name: int, review_aroma: float, review_appearance: float, review_palate: float, review_taste: float):
    
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

    # answer_dict = {k: [v] for (k, v) in jsonable_encoder(answer).items()}
    return JSONResponse(answer.tolist())

@app.post("/beers/type/")
async def predict_beer_style_multi(
        brewery_name: Optional[List[int]] = Query(...,min_value=1),
        review_aroma: Optional[List[float]] = Query(...,min_value=1),
        review_appearance: Optional[List[float]] = Query(...,min_value=1),
        review_palate: Optional[List[float]] = Query(...,min_value=1),
        review_taste: Optional[List[float]] = Query(...,min_value=1),):

    query_items = {"brewery_name": brewery_name,
                   "review_aroma": review_aroma,
                   'review_appearance': review_appearance,
                   "review_palate": review_palate,
                   "review_taste": review_taste}

    features = format_features_multi(brewery_name,review_aroma, review_appearance, review_palate, review_taste)
    obs = pd.DataFrame(features)
    # scale num cols
    obs = apply_scaler(obs)
    
    #transform to embed object
    obs_tensor = EmbeddingDataset(obs)
    
    #predict on embed obj
    model = get_model()
    
    #return predictions as text string
    answer = predict(obs_tensor, model, single=False)

    #answer_dict = {k: [v] for (k, v) in jsonable_encoder(answer.tolist()).items()}
    return JSONResponse(answer.tolist())


@app.get("/model/architecture/")
async def print_model():
    
    model = get_model()

    return {'model archtecture': str(model)}


