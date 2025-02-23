import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.ops

# from https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/8

SMOTE_LABEL = 1

    
class CappedBCELoss(nn.Module):
    def __init__(self, loss_cap=None, distance=None, reduction='mean', cap_array=None, function=None, print_loss=False, print_capped=False):
        nn.Module.__init__(self)
        self.loss_cap = loss_cap # for constant cap, or hyperparameter to multiply cap by 
        self.distance = distance # for cosine or euclidean distance to be used 
        self.reduction = reduction
        self.cap_array = cap_array # used if cap already calculated (e.g. using triplet loss as the cap)
        self.function = function # for squaring, etc. 
        self.print_loss = print_loss
        self.print_capped = print_capped
        
    def cosine_distance(self, data, targets, smote_targets):
        dist = torch.zeros(data.shape[0])
        targets = np.array(targets)
        for label in range(2):
            mask = np.zeros(targets.shape)
            mask[targets == label] = 1
            mask[smote_targets == SMOTE_LABEL] = 0
            avg = torch.mean(data[mask == 1], 0)
            dist[targets == label] = 1 - F.cosine_similarity(data[targets == label], avg)
        return dist

    def euclidean_distance(self, data, targets, smote_targets):
        dist = torch.zeros(data.shape[0])
        targets = np.array(targets)
        for label in range(2):
            mask = np.zeros(targets.shape)
            mask[targets == label] = 1
            mask[smote_targets == SMOTE_LABEL] = 0
            avg = torch.mean(data[mask == 1], 0)
            dist[targets == label] = (data[targets == label] - avg).pow(2).sum(
                1).sqrt()
        return dist
        
        
    def forward(self, inputs, targets, smote_targets, embeds=None):
        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        if self.print_loss:
            print("Loss before cap") 
            print(loss) 
            print("SMOTE Labels") 
            print(smote_targets) 
        if self.loss_cap != None:
            if self.cap_array != None:
                # capped smote using triplet loss
                loss[smote_targets == SMOTE_LABEL] = torch.minimum(loss[smote_targets == SMOTE_LABEL], torch.tensor(self.cap_array * self.loss_cap)[smote_targets==SMOTE_LABEL])
            elif self.distance != None:
                # always pass in embeddings  
                if self.distance == 'euclidean':
                    distances = self.euclidean_distance(embeds, targets, smote_targets)
                elif self.distance == 'cosine':
                    distances = self.cosine_distance(embeds, targets, smote_targets) 
                if self.function == 'square':
                    cap = ((1 / distances) ** 2) * self.loss_cap
                else:
                    cap = self.loss_cap / distances
                if self.print_capped:
                    loss_mask = torch.zeros(loss.shape, dtype=torch.int) 
                    loss_mask[smote_targets == SMOTE_LABEL] = 1
                    loss_mask[loss < cap] = 0
                    n_capped = torch.sum(loss_mask) 
                    amt_capped = torch.sum(loss[loss_mask]) 
                    print("Number capped: " + str(n_capped.item())) 
                    print("Average cap: " + str((amt_capped/n_capped).item())) 
                loss[smote_targets == SMOTE_LABEL] = torch.minimum(loss[smote_targets == SMOTE_LABEL], torch.tensor(cap)[smote_targets==SMOTE_LABEL])
            else:
                if self.print_capped:
                    loss_mask = torch.zeros(loss.shape, dtype=torch.int) 
                    loss_mask[smote_targets == SMOTE_LABEL] = 1
                    loss_mask[loss < self.loss_cap] = 0
                    n_capped = torch.sum(loss_mask) 
                    amt_capped = torch.sum(loss[loss_mask]) 
                    print("Number capped: " + str(n_capped.item())) 
                    print("Average cap: " + str((amt_capped/n_capped).item())) 
                # cap is a constant 
                loss[smote_targets == SMOTE_LABEL] = torch.minimum(loss[smote_targets == SMOTE_LABEL], torch.tensor(self.loss_cap))
                
        if self.reduction=='mean':
            return torch.mean(loss)
        return loss
    
        
class CappedCELoss(nn.Module):
    def __init__(self, loss_cap=None, distance=None, reduction='mean', cap_array=None, function=None, print_capped=False):
        nn.Module.__init__(self)
        self.loss_cap = loss_cap # for constant cap, or hyperparameter to multiply cap by 
        self.distance = distance # for cosine or euclidean distance to be used 
        self.reduction = reduction
        self.cap_array = cap_array # used if cap already calculated (e.g. using triplet loss)
        self.function = function # for squaring, etc. 
        self.print_capped = print_capped 
        
    def cosine_distance(self, data, targets, smote_targets):
        dist = torch.zeros(data.shape[0])
        targets = np.array(targets)
        for label in range(len(np.unique(targets))):
            mask = np.zeros(targets.shape)
            mask[targets == label] = 1
            mask[smote_targets == SMOTE_LABEL] = 0
            avg = torch.mean(data[mask == 1], 0)
            dist[targets == label] = 1 - F.cosine_similarity(data[targets == label], avg)
        return dist

    def euclidean_distance(self, data, targets, smote_targets):
        dist = torch.zeros(data.shape[0])
        targets = np.array(targets)
        for label in range(len(np.unique(targets))):
            mask = np.zeros(targets.shape)
            mask[targets == label] = 1
            mask[smote_targets == SMOTE_LABEL] = 0
            avg = torch.mean(data[mask == 1], 0)
            dist[targets == label] = (data[targets == label] - avg).pow(2).sum(
                1).sqrt()
        return dist
        
        
    def forward(self, inputs, targets, smote_targets, embeds=None):
        loss = F.cross_entropy(inputs, targets.long(), reduction='none')
        if self.print_capped:
            loss_mask = torch.zeros(loss.shape, dtype=torch.int) 
            loss_mask[smote_targets == SMOTE_LABEL] = 1
            loss_mask[loss < self.loss_cap] = 0
            n_capped = torch.sum(loss_mask) 
            amt_capped = torch.sum(loss[loss_mask]) 
            print("Number capped: " + str(n_capped.item())) 
            print("Average cap: " + str((amt_capped/n_capped).item())) 
        if self.loss_cap != None:
            if self.cap_array != None:
                # capped smote using triplet loss
                loss[smote_targets == SMOTE_LABEL] = torch.minimum(loss[smote_targets == SMOTE_LABEL], torch.tensor(self.cap_array * self.loss_cap)[smote_targets==SMOTE_LABEL])
            elif self.distance != None:
                if self.distance == 'euclidean':
                    distances = self.euclidean_distance(embeds, targets, smote_targets)
                elif self.distance == 'cosine':
                    distances = self.cosine_distance(embeds, targets, smote_targets) 
                if self.function == 'square':
                    cap = (1 / distances) ** 2 * self.loss_cap
                else:
                    cap = self.loss_cap / distances
                loss[smote_targets == SMOTE_LABEL] = torch.minimum(loss[smote_targets == SMOTE_LABEL], torch.tensor(cap)[smote_targets==SMOTE_LABEL])
            else:
                # cap is a constant 
                loss[smote_targets == SMOTE_LABEL] = torch.minimum(loss[smote_targets == SMOTE_LABEL], torch.tensor(self.loss_cap))
        if self.reduction=='mean':
            return torch.mean(loss)
        return loss

class CappedBCELossAvgDistance(nn.Module): 
    def __init__(self, loss_cap=None, distance=None, avg_tensors=None, reduction='mean', function=None, print_loss=False):
        nn.Module.__init__(self)
        self.loss_cap = loss_cap # for constant cap, or hyperparameter to multiply cap by 
        self.distance = distance # for cosine or euclidean distance to be used 
        self.reduction = reduction
        self.avg_tensors = avg_tensors
        self.function = function # for squaring, etc. 
        self.print_loss = print_loss
        
    def cosine_distance(self, data, targets, smote_targets):
        dist = torch.zeros(data.shape[0])
        targets = np.array(targets)
        for label in range(2):
            dist[targets == label] = 1 - F.cosine_similarity(data[targets == label], self.avg_tensors[label])
        return dist

    def euclidean_distance(self, data, targets, smote_targets):
        dist = torch.zeros(data.shape[0])
        targets = np.array(targets)
        for label in range(2):
            dist[targets == label] = (data[targets == label] - self.avg_tensors[label]).pow(2).sum(
                1).sqrt()
        return dist
        
        
    def forward(self, inputs, targets, smote_targets, embeds=None):
        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        if self.loss_cap != None:
            # always pass in embeddings  
            if self.distance == 'euclidean':
                distances = self.euclidean_distance(embeds, targets, smote_targets)
            elif self.distance == 'cosine':
                distances = self.cosine_distance(embeds, targets, smote_targets) 
            if self.function == 'square':
                cap = ((1 / distances) ** 2) * self.loss_cap
            else:
                cap = self.loss_cap / distances
            if self.print_loss:
                print("Loss before cap") 
                print(loss) 
                print("SMOTE Labels") 
                print(smote_targets) 
                print("Distance") 
                print(distances) 
            
            loss[smote_targets == SMOTE_LABEL] = torch.minimum(loss[smote_targets == SMOTE_LABEL], torch.tensor(cap[smote_targets==SMOTE_LABEL]))
        if self.reduction=='mean':
            return torch.mean(loss)
        return loss
    

class CappedCELossAvgDistance(nn.Module): 
    def __init__(self, loss_cap=None, distance=None, avg_tensors=None, reduction='mean', function=None, print_loss=False):
        nn.Module.__init__(self)
        self.loss_cap = loss_cap # for constant cap, or hyperparameter to multiply cap by 
        self.distance = distance # for cosine or euclidean distance to be used 
        self.reduction = reduction
        self.avg_tensors = avg_tensors
        self.function = function # for squaring, etc. 
        self.print_loss = print_loss
        
    def cosine_distance(self, data, targets, smote_targets):
        dist = torch.zeros(data.shape[0])
        targets = np.array(targets)
        for label in range(len(np.unique(targets))):
            dist[targets == label] = 1 - F.cosine_similarity(data[targets == label], self.avg_tensors[label])
        return dist

    def euclidean_distance(self, data, targets, smote_targets):
        dist = torch.zeros(data.shape[0])
        targets = np.array(targets)
        for label in range(len(np.unique(targets))):
            dist[targets == label] = (data[targets == label] - self.avg_tensors[label]).pow(2).sum(
                1).sqrt()
        return dist
        
        
    def forward(self, inputs, targets, smote_targets, embeds=None):
            loss = F.cross_entropy(inputs, targets.long(), reduction='none')       
            if self.loss_cap != None:
            # always pass in embeddings  
                if self.distance == 'euclidean':
                    distances = self.euclidean_distance(embeds, targets, smote_targets)
                elif self.distance == 'cosine':
                    distances = self.cosine_distance(embeds, targets, smote_targets) 
                if self.function == 'square':
                    cap = ((1 / distances) ** 2) * self.loss_cap
                else:
                    cap = self.loss_cap / distances
                if self.print_loss:
                    print("Loss before cap") 
                    print(loss) 
                    print("SMOTE Labels") 
                    print(smote_targets) 
                    print("Distance") 
                    print(distances) 

                loss[smote_targets == SMOTE_LABEL] = torch.minimum(loss[smote_targets == SMOTE_LABEL], torch.tensor(cap[smote_targets==SMOTE_LABEL]))
            if self.reduction=='mean':
                return torch.mean(loss)
            return loss
    
    
class TripletLoss(nn.Module):
    def __init__(self, margin=0.3, reduction='mean'):
        super(TripletLoss, self).__init__()
       # self.margin = margin
       # self.reduction = reduction
        self.tripletloss = nn.TripletMarginLoss(margin=margin, p=2.0, swap=False, reduction=reduction)
        
    def euclidean_distance(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.euclidean_distance(anchor, positive)
        distance_negative = self.euclidean_distance(anchor, negative)
        losses = self.tripletloss(anchor, positive, negative) 
        return losses
    
    
class TripletLossWithAverage(nn.Module):
    def __init__(self, margin=0.3, reduction='mean'):
        super(TripletLossWithAverage, self).__init__()
        self.margin = margin
        self.reduction = reduction
        
    def euclidean_distance(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)
    
    def get_average(self, x):
        # returns average tensor of examples of pos/neg class in batch 
        return torch.mean(x, 0)
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        avg_positive = self.get_average(positive)
        avg_negative = self.get_average(negative)
        distance_positive = self.euclidean_distance(anchor, avg_positive)
        distance_negative = self.euclidean_distance(anchor, avg_negative)
        if self.reduction == 'mean':
            distance_positive = distance_positive.mean()
            distance_negative = distance_negative.mean()
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        losses[losses == 0] = 0.00001
        return losses
    
    
class TripletLossSquaredPositive(nn.Module):
    def __init__(self, margin=0.3, reduction='mean'):
        super(TripletLossSquaredPositive, self).__init__()
        self.margin = margin
        self.reduction = reduction
        
    def euclidean_distance(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.euclidean_distance(anchor, positive)
        distance_positive = distance_positive**2 
        distance_negative = self.euclidean_distance(anchor, negative)
        if self.reduction == 'mean':
            distance_positive = distance_positive.mean()
            distance_negative = distance_negative.mean()
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        losses[losses == 0] = 0.00001
        return losses
    
    

class SigmoidFocalLoss(nn.Module):
    def __init__(self, weight=None, 
                 gamma=2., reduction='none'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        return torchvision.ops.sigmoid_focal_loss(inputs * self.weight, targets, gamma=self.gamma, reduction=self.reduction)
    

class SoftmaxFocalLoss(nn.Module): 
    def __init__(self, weight=None, 
                 gamma=2., reduction='none'):
        nn.Module.__init__(self)
        self.alpha = weight
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        log_prob = F.log_softmax(inputs, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob, 
            targets, 
            weight=self.alpha,
            reduction = self.reduction)
    
    