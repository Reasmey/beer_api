# model script to load to main.py

import numpy as np
import pandas as pd
import joblib
from sklearn.utils import StandardScaler
import torch
from torch.utils.data import Dataset


# formater for single input
def format_features(brewery_name: int, review_aroma: int, review_appearance: int, review_palate: int, review_taste: int):
      
        return {
        'brewery_name': [brewery_name],
        'review_aroma': [review_aroma],
        'review_appearance': [review_appearance],
        'review_palate': [review_palate],
        'review_taste': [review_taste]
    }


# load scaler
def apply_scaler(obs):
    '''Applies scaler to num_cols'''
    
    num_cols = ['review_aroma','review_appearance','review_palate','review_taste']
    scaler = load('../models/scaler.joblib')
    obs[num_cols] = scaler.fit_transform(obs[num_cols])
    
    return obs

# create tensors for single input
def single_tensor(obs):
    """Converts single row to tensor """
    data_cat = []
    data_cont = []
    num_cols = ['review_aroma','review_appearance','review_palate','review_taste']
    data_cat = torch.tensor(obs['brewery_name'].to_numpy())
    data_cont = torch.tensor(obs[num_cols].to_numpy())
                
    data = [data_cat, data_cont]
    result = {'data': data}
    
    return result

# Dataset creator for embedding for multi input
class EmbeddingDataset(Dataset):
    def __init__(self, data, targets=None,
                 is_train=True, cat_cols_idx=None,
                 cont_cols_idx=None):
        self.data = data
        self.targets = targets
        self.is_train = is_train
        self.cat_cols_idx = cat_cols_idx
        self.cont_cols_idx = cont_cols_idx
    
    def __getitem__(self, idx):
        row = self.data[idx].astype('float32')
        
        data_cat = []
        data_cont = []
        
        result = None
        
        if self.cat_cols_idx:
            data_cat = torch.tensor(row[self.cat_cols_idx])
            
        if self.cont_cols_idx:
            data_cont = torch.tensor(row[self.cont_cols_idx])
                
        data = [data_cat, data_cont]
                
        if self.is_train:
            result = {'data': data,
                      'target': torch.tensor(self.targets[idx])}
        else:
            result = {'data': data}
            
        return result
            
    
    def __len__(self):
        return(len(self.data))
    
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
    
# get model and load
def get_model():
    
    # load model obj
    model = torch.load('../models/model.pt')
    #set to trained dict of weights
    model.load_state_dict(torch.load('../models/embed_3layers.pt'))

    return model
    
# function to predict from obs
def predict(obs, model, single=False):
    """obs = dataset as tensor embed obj
       model = model state_dict loaded
       single = set to true if single input"""
    
    # set to eval
    model.eval()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    
    with torch.no_grad():
        predictions = None
        
        if single:
            output = model(obs['data'][0].unsqueeze(dim=0).to(device, 
                                     dtype=torch.long), 
                       obs['data'][1].to(device, 
                                     dtype=torch.float)).cpu().numpy()
        else:
            for i, batch in enumerate(obs):   
            
                output = model(batch['data'][0].to(device, 
                                               dtype=torch.long), 
                               batch['data'][1].to(device, 
                                               dtype=torch.float)).cpu().numpy()
            
                if i == 0:
                    predictions = output
                
                else: 
                
                    predictions = np.vstack((predictions, output))   

        predictions = output
     
    from joblib import load
    label_encoders = load('../models/label_encoders.joblib')
    label = label_encoders['beer_style'].inverse_transform(predictions.argmax(1))
    
          
    return label
    

