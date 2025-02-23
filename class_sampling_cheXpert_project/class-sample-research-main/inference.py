import torch
import torch.nn as nn
import torch.nn.functional as F
import loss_fns

import numpy as np

def run_inference_triplet_loss(dataloader, network):
    losses = []
    loss = 0
    
    network.eval()
    
    loss_fn=loss_fns.TripletLoss(margin=1000); 
    
    with torch.no_grad():
        for anchor_data, pos_data, neg_data, target in dataloader:
            anchor_embed = network(anchor_data)
            pos_embed = network(pos_data)
            neg_embed = network(neg_data)
            loss += loss_fn(anchor_embed, pos_embed, neg_embed).item()
        loss /= len(dataloader)
        losses.append(loss)
    
    return losses
    

def run_inference_sigmoid(dataloader, network, embeddings=False):  
    losses = []
    y_preds = []
    y_true = []
    
    network.eval()
    loss = 0
    correct = 0
    
    loss_fn=nn.BCEWithLogitsLoss()
    
    with torch.no_grad():
        for data, target in dataloader:
          #print(data.shape)
          output = network(data)
          print(output.shape)
          if embeddings:
            output = output[0]
          loss += loss_fn(output.squeeze().float(), target.float()).item()
          pred=torch.sigmoid(output.data) # probabilities
          y_preds.extend(pred.float())
          y_true.extend(target.float())
        loss /= len(dataloader.dataset)
        losses.append(loss)
    
    return losses, y_preds, y_true

def run_inference_softmax(dataloader, network, embeddings=False): 
    losses = []
    y_preds = None
    y_true = None
    
    network.eval()
    loss = 0
    correct = 0
    
    loss_fn=nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data, target in dataloader:
            output = network(data)
            if embeddings: 
                output = output[0]
            loss += loss_fn(output.squeeze(), target).item()
           # pred = output.data.max(1, keepdim=True)[1]
            pred = F.softmax(output.data)
            if y_preds is None:
                y_preds = torch.tensor(pred)
                y_true = torch.tensor(target)
            else:
                y_preds = np.concatenate((y_preds, pred), axis=0)
                y_true = np.concatenate((y_true, target), axis=0)
                
        loss /= len(dataloader.dataset)
        losses.append(loss)
        
    return losses, y_preds, y_true
