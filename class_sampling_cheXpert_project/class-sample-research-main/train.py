import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import loss_fns
import numpy as np
import loss_fns

LOG_INTERVAL = 1000


def train_sigmoid(epoch, train_loader, network, optimizer, directory=None,
                  verbose=True, loss_fn=nn.BCEWithLogitsLoss, loss_fn_args={}):
    # train for binary classification. No SMOTE. 
    train_counter = []
    train_losses = []

    loss_fn = loss_fn(**loss_fn_args)

    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        if type(output) is tuple:
            output = output[0] 
        loss = loss_fn(output.squeeze().float(), target.float())
        pred = output.data
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0 and verbose:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                         batch_idx * len(data),
                                                                         len(train_loader.dataset),
                                                                         100. * batch_idx /
                                                                         len(train_loader),
                                                                         loss.item()))
        train_losses.append(loss.item())
        train_counter.append(
            (batch_idx * 64) + ((epoch) * len(train_loader.dataset)))
    if directory:
        torch.save(network.state_dict(), directory)
    return train_counter, train_losses


def train_sigmoid_with_smote(epoch, train_loader, network, optimizer,
                             directory=None, verbose=True,
                             loss_fn=loss_fns.CappedBCELoss, loss_fn_args={}):
    # train for binary classification with SMOTE. 
    
    train_counter = []
    train_losses = []

    loss_fn = loss_fn(**loss_fn_args)

    network.train()
    for batch_idx, (data, target, smote_target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = loss_fn(output.squeeze().float(), target.float(), smote_target)
        pred = output.data
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0 and verbose:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                         batch_idx * len(
                                                                             data),
                                                                         len(train_loader.dataset),
                                                                         100. * batch_idx /
                                                                         len(train_loader),
                                                                         loss.item()))
        train_losses.append(loss.item())
        train_counter.append(
            (batch_idx * 64) + ((epoch) * len(train_loader.dataset)))
    if directory:
        torch.save(network.state_dict(), directory)
    return train_counter, train_losses

def train_sigmoid_with_embeddings(epoch, train_loader, network, optimizer,
                                  directory=None, verbose=True,
                                  loss_fn=loss_fns.CappedBCELoss,
                                  loss_fn_args={}):
    # always uses SMOTE

    train_counter = []
    train_losses = []

    loss_func = loss_fn(**loss_fn_args)

    network.train()
    for batch_idx, (data, target, smote_target) in enumerate(train_loader):
        optimizer.zero_grad()

        output, embeds = network(data)
        
        loss = loss_func(output.squeeze().float(), target.float(), smote_target, embeds=embeds)
        pred = output.data
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0 and verbose:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                         batch_idx * len(
                                                                             data),
                                                                         len(train_loader.dataset),
                                                                         100. * batch_idx /
                                                                         len(train_loader),
                                                                         loss.item()))
        train_losses.append(loss.item())
        train_counter.append(
            (batch_idx * 64) + ((epoch) * len(train_loader.dataset)))
    if directory:
        torch.save(network.state_dict(), directory)
    return train_counter, train_losses

def train_triplet_loss(epoch, train_loader, network, optimizer, directory=None,
                       verbose=True, loss_fn_args={}):
    # used for triplet loss training (binary or multiclass). No SMOTE. 
    train_counter = []
    train_losses = []

    loss_fn = loss_fns.TripletLoss(**loss_fn_args)
   

    network.train()
    for batch_idx, (anchor_data, pos_data, neg_data, target) in enumerate(
            train_loader):
        optimizer.zero_grad()
        anchor_embeds = network(anchor_data)
        pos_embeds = network(pos_data)
        neg_embeds = network(neg_data)
        loss = loss_fn(anchor_embeds, pos_embeds, neg_embeds)
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0 and verbose:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                         batch_idx * len(
                                                                             anchor_data),
                                                                         len(train_loader.dataset),
                                                                         100. * batch_idx / len(
                                                                             train_loader),
                                                                         loss.item()))
        train_losses.append(loss.item())
        train_counter.append(
            (batch_idx * 64) + ((epoch) * len(train_loader.dataset)))
    if directory:
        torch.save(network.state_dict(), directory)
    return train_counter, train_losses



def train_triplet_loss_smote(epoch, train_loader, network, optimizer, directory=None,
                       verbose=True, loss_fn_args={}):
    # used for triplet loss training (binary or multiclass) on datasets using SMOTE. 
    
    train_counter = []
    train_losses = []

    loss_fn = loss_fns.TripletLoss(**loss_fn_args)
   

    network.train()
    for batch_idx, (anchor_data, pos_data, neg_data, target, smote_target) in enumerate(
            train_loader):
        optimizer.zero_grad()
        anchor_embeds = network(anchor_data)
        pos_embeds = network(pos_data)
        neg_embeds = network(neg_data)
        loss = loss_fn(anchor_embeds, pos_embeds, neg_embeds)
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0 and verbose:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                         batch_idx * len(
                                                                             anchor_data),
                                                                         len(train_loader.dataset),
                                                                         100. * batch_idx / len(
                                                                             train_loader),
                                                                         loss.item()))
        train_losses.append(loss.item())
        train_counter.append(
            (batch_idx * 64) + ((epoch) * len(train_loader.dataset)))
    if directory:
        torch.save(network.state_dict(), directory)
    return train_counter, train_losses

def train_linear_probe(epoch, train_loader, network,
                       optimizer, directory=None, verbose=True,
                       loss_fn=nn.BCEWithLogitsLoss, loss_fn_args={}):
    
    train_counter = []
    train_losses = []

    loss_fn = loss_fn(**loss_fn_args)

    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()  
        output, _ = network(data)
        loss = loss_fn(output.squeeze().float(), target.float())
        pred = output.data
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0 and verbose:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                         batch_idx * len(
                                                                             data),
                                                                         len(train_loader.dataset),
                                                                         100. * batch_idx /
                                                                         len(train_loader),
                                                                         loss.item()))
        train_losses.append(loss.item())
        train_counter.append(
            (batch_idx * 64) + ((epoch) * len(train_loader.dataset)))
    if directory:
        torch.save(network.state_dict(), directory)
    return train_counter, train_losses


def train_triplet_capped_loss(epoch, train_loader, network, optimizer, directory=None,
                       verbose=True, cap_calc=loss_fns.TripletLoss, loss_fn=loss_fns.CappedBCELoss, loss_fn_args={}, print_dist=False):
    
    # used for training capped loss (on SMOTE, both binary and multiclass) using the triplet loss function as the cap.
    
    train_counter = []
    train_losses = []

    cap_calc = cap_calc(reduction='none')
        
    loss_func = loss_fn(**loss_fn_args)
    

    network.train()
    for batch_idx, (anchor_data, pos_data, neg_data, target, smote_target) in enumerate(
            train_loader):
        optimizer.zero_grad()
        
        anchor_output, anchor_embeds = network(anchor_data.float())
        _, pos_embeds = network(pos_data.float())
        _, neg_embeds = network(neg_data.float())
        
        
        cap = cap_calc(anchor_embeds, pos_embeds, neg_embeds) 
        
        if print_dist:
            print("Triplet Loss Calculation") 
            print(cap) 
        
        loss_fn_args['cap_array'] = 1 / cap # cap is inverse of distances 
        loss_func = loss_fn(**loss_fn_args)
        
        
        loss = loss_func(anchor_output.squeeze(), target.float(), smote_target)
        
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0 and verbose:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                         batch_idx * len(
                                                                             anchor_data),
                                                                         len(train_loader.dataset),
                                                                         100. * batch_idx / len(
                                                                             train_loader),
                                                                         loss.item()))
        train_losses.append(loss.item())
        train_counter.append(
            (batch_idx * 64) + ((epoch) * len(train_loader.dataset)))
    if directory:
        torch.save(network.state_dict(), directory)
    
    return train_counter, train_losses


def train_softmax(epoch, train_loader, network, optimizer, directory=None,
                  verbose=True, loss_fn=nn.CrossEntropyLoss, loss_fn_args={}):
    # train for multiclass classification. No SMOTE. 
    train_counter = []
    train_losses = []

    network.train()

    loss_fn = loss_fn(**loss_fn_args)

    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = loss_fn(output.squeeze(), target.long())
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0 and verbose:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                         batch_idx * len(
                                                                             data),
                                                                         len(train_loader.dataset),
                                                                         100. * batch_idx /
                                                                         len(train_loader),
                                                                         loss.item()))
        train_losses.append(loss.item())
        train_counter.append(
            (batch_idx * 64) + ((epoch) * len(train_loader.dataset)))
    if directory:
        torch.save(network.state_dict(), directory)
    return train_counter, train_losses


def train_softmax_with_smote(epoch, train_loader, network, optimizer,
                             directory=None, verbose=True,
                             loss_fn=loss_fns.CappedCELoss, loss_fn_args={}):
    # train for multiclass classification with SMOTE. 
    train_counter = []
    train_losses = []

    network.train()

    loss_fn = loss_fn(**loss_fn_args)

    for batch_idx, (data, target, smote_target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = loss_fn(output.squeeze(), target, smote_target)
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0 and verbose:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                         batch_idx * len(
                                                                             data),
                                                                         len(train_loader.dataset),
                                                                         100. * batch_idx /
                                                                         len(train_loader),
                                                                         loss.item()))
        train_losses.append(loss.item())
        train_counter.append(
            (batch_idx * 64) + ((epoch) * len(train_loader.dataset)))
    if directory:
        torch.save(network.state_dict(), directory)
    return train_counter, train_losses
    
def train_softmax_with_embeddings(epoch, train_loader, network, optimizer,
                             directory=None, verbose=True,
                             loss_fn=loss_fns.CappedCELoss, loss_fn_args={}):
    train_counter = []
    train_losses = []

    network.train()

    loss_fn = loss_fn(**loss_fn_args)

    for batch_idx, (data, target, smote_target) in enumerate(train_loader):
        optimizer.zero_grad()
        output, embeds = network(data)
        loss = loss_fn(output.squeeze(), target, smote_target, embeds=embeds)
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0 and verbose:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                         batch_idx * len(
                                                                             data),
                                                                         len(train_loader.dataset),
                                                                         100. * batch_idx /
                                                                         len(train_loader),
                                                                         loss.item()))
        train_losses.append(loss.item())
        train_counter.append(
            (batch_idx * 64) + ((epoch) * len(train_loader.dataset)))
    if directory:
        torch.save(network.state_dict(), directory)
    return train_counter, train_losses


def train_sigmoid_with_smote_embeddings(epoch, train_loader, network, optimizer,
                                  directory=None, verbose=True,
                                  loss_fn=loss_fns.CappedBCELoss,
                                  loss_fn_args={}):
    
    # uses smote on embeddings - need to pass into function 

    train_counter = []
    train_losses = []

    loss_func = loss_fn(**loss_fn_args)

    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output, embeds = network(data)
        loss = loss_func(output.squeeze().float(), target.float(), smote_target, embeds=embeds)
        pred = output.data
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0 and verbose:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                         batch_idx * len(
                                                                             data),
                                                                         len(train_loader.dataset),
                                                                         100. * batch_idx /
                                                                         len(train_loader),
                                                                         loss.item()))
        train_losses.append(loss.item())
        train_counter.append(
            (batch_idx * 64) + ((epoch) * len(train_loader.dataset)))
    if directory:
        torch.save(network.state_dict(), directory)
    return train_counter, train_losses