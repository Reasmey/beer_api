from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
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

@app.get("/",response_class=HTMLResponse)
def read_root():
    return """
    <html>
        <head>
            <title>Predicting a style of beer according to ratings of taste, aroma, appearance and palate scores</title>
        </head>
        <body>
            <h1>Endpoints</h1>
            <p>
            ‘/’ (GET): Displaying a brief description of the project objectives expected input parameters and output format of the model, link to the Github repo related to this project <br>
            '/docs/': Summary of all endpoints. <br>
            ‘/health/’ (GET): Returning status code 200 with a welcome message <br>
            ‘/beer/type/’ (POST): Returning prediction for a single input <br>
            ‘/beers/type/’ (POST): Returning predictions for a multiple inputs <br>
            ‘/model/architecture/’ (GET): Displaying the architecture of your Neural Networks <br>
            </p>
            <h1>Instructions</h1>
            <p>
            <h2>Input parameters to API</h2>
            brewery_name : integer between 1-5742, please see the list of brewery names with their integer ID's at https://github.com/Reasmey/adsi_beer_app/blob/api_checks/data/processed/Brewery_Encode.csv <br>
            review_aroma : float between 1.0 and 5.0 <br>
            review_appearance : float between 0.0 and 5.0 <br>
            review_palate : float between 1.0 and 5.0 <br>
            review_taste : float between 1.0 and 5.0 <br>
            <br>
            For beer/type/, one value for each parameter is expected as input. <br>
            For beers/type/, you can input multiple values for each parameter, but you need to put the same number of values for each parameter, and please not too many. 
            </p>
            <h2>Outputs expected</h2>
            <p>
            For beer/type/, a string should be returned of the style of beer that was predicted from the single parameter inputs. <br>
            For beers/type/, a string should be returned for each input to the parameters, e.g. 3 inputs per parameter = 3 predictions returned. <br>
            </p>
            <h1>links to the Github repos</h1>
            <p>
            https://github.com/Reasmey/beer_api <br>
            https://github.com/Reasmey/adsi_beer_app
            </p>"""



@app.get('/health', status_code=200)
def healthcheck():
    return 'Glug glug glug'


@app.post("/beer/type/")
async def predict_beer_style(brewery_name: int, review_aroma: float, review_appearance: float, review_palate: float, review_taste: float):
    
    features = format_features(brewery_name,review_aroma, review_appearance, review_palate, review_taste)
    obs = pd.DataFrame(features)
    # scale num cols
    obs = apply_scaler(obs)

    # encode cat col brewery name
    obs = apply_name_encoder(obs)
    
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
        brewery_name: Optional[List[int]] = Query(...,min_value=1, max_value=5742),
        review_aroma: Optional[List[float]] = Query(...,min_value=1.0, max_value=5.0),
        review_appearance: Optional[List[float]] = Query(...,min_value=0.0, max_value=5.0),
        review_palate: Optional[List[float]] = Query(...,min_value=1.0, max_value=5.0),
        review_taste: Optional[List[float]] = Query(...,min_value=1.0, max_value=5.0),):

    features = format_features_multi(brewery_name,review_aroma, review_appearance, review_palate, review_taste)
    obs = pd.DataFrame(features)
    # scale num cols
    obs = apply_scaler(obs)

    # encode cat col brewery name
    obs = apply_name_encoder(obs)
    
    #transform to embed object
    obs_tensor = EmbeddingDataset(obs.to_numpy(),
                             cat_cols_idx=[0],
                             cont_cols_idx=[1,2,3,4],
                             is_train=False)
    
    #predict on embed obj
    model = get_model()
    
    #return predictions as text string
    answer = predict(obs_tensor, model, single=False)

    #answer_dict = {k: [v] for (k, v) in jsonable_encoder(answer.tolist()).items()}
    return JSONResponse(answer.tolist())


@app.get("/model/architecture/", response_class=HTMLResponse)
def print_model():
    
    return """
        <html>
            <head>
                <title>Model Architecture</title>
            </head>
            <body>
                ClassificationEmbdNN( <br>
                  (emb_layers): ModuleList( <br>
                    (0): Embedding(5742, 252) <br>
                  ) <br>
                  (emb_dropout): Dropout(p=0.2, inplace=False) <br>
                  (bn_cont): BatchNorm1d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) <br>
                  (fc1): Linear(in_features=256, out_features=208, bias=True) <br>
                  (dropout1): Dropout(p=0.2, inplace=False) <br>
                  (bn1): BatchNorm1d(208, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) <br>
                  (act1): ReLU() <br>
                  (fc2): Linear(in_features=208, out_features=208, bias=True) <br>
                  (dropout2): Dropout(p=0.2, inplace=False) <br>
                  (bn2): BatchNorm1d(208, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) <br>
                  (act2): ReLU() <br>
                  (fc3): Linear(in_features=208, out_features=104, bias=True) <br>
                  (act3): Softmax(dim=None) <br>
                )
            </body>
        </html>
        """


