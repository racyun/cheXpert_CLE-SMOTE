import torch 
import torch.nn as nn
import torch.nn.functional as F


class GrayscaleImage(nn.Module):
  '''
  Used to tile grayscale matrices to convert to 3-channel
  '''
  def __init__(self, model):
      super(GrayscaleImage, self).__init__()
      self.model = model

  def forward(self, x):
      x = x.repeat(1, 3, 1, 1)
      return self.model(x)


class Net(nn.Module):
  '''
  3 linear layers. When called, adds these three linear layers to end of model.
  '''
  def __init__(self):
      super(Net, self).__init__()
      self.fc1 = nn.Linear(1000, 512)
      self.fc2 = nn.Linear(512, 256)
      self.fc3 = nn.Linear(256, 1)

  def forward(self, x):
      x = self.fc1(x)
      x = self.fc2(x)
      x = self.fc3(x)
      return x


class ConvNet(nn.Module):
    def __init__(self, num_classes):
        print("hello", flush=True)
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 10, kernel_size=5)
        self.conv1_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(250, 50) 
        if (num_classes == 2): 
            self.fc2 = nn.Linear(50, 1) 
        else:
            self.fc2 = nn.Linear(50, num_classes)
            

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1_drop(self.conv1(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 250) 
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

class ConvNet_grayscale(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet_grayscale, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 10, kernel_size=5)
        self.conv1_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(28090, 1000) 
        if (num_classes == 2): 
            self.fc2 = nn.Linear(1000, 1) 
        else:
            self.fc2 = nn.Linear(50, num_classes)
            

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1_drop(self.conv1(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 28090) 
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

class ConvNetWithEmbeddings(nn.Module):
    def __init__(self, num_classes):
        super(ConvNetWithEmbeddings, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 10, kernel_size=5)
        self.conv1_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(250, 50)
        if (num_classes == 2): 
            self.fc2 = nn.Linear(50, 1) 
        else:
            self.fc2 = nn.Linear(50, num_classes)


    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1_drop(self.conv1(x)), 2))
        x = F.relu(F.max_pool2d((self.conv2(x)), 2))
        x = x.view(-1, 250) 
        embed = self.fc1(x)
        x = F.relu(F.dropout(embed, training=self.training))
        x = self.fc2(x)
        return x, embed
    
class ConvNetOnlyEmbeddings(nn.Module):
    def __init__(self, num_classes):
        super(ConvNetOnlyEmbeddings, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 10, kernel_size=5)
        self.conv1_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(250, 50)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1_drop(self.conv1(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 250) 
        embed = self.fc1(x) 
        return embed

class ConvNetLinearProbe(nn.Module): 
    def __init__(self, num_classes):
        super(ConvNetLinearProbe, self).__init__()
        if (num_classes == 2): 
            self.fc2 = nn.Linear(50, 1) 
        else:
            self.fc2 = nn.Linear(50, num_classes)
            
    def forward(self, embed):
        x =  F.relu(F.dropout(embed, training=self.training)) 
        x = self.fc2(x)
        return x
    

class CompleteConvNet(nn.Module): 
    def __init__(self, embed_network, linear_probe):
        super(CompleteConvNet, self).__init__()
        self.embed_network = embed_network
        self.linear_probe = linear_probe
        
    def forward(self, x): 
        embeds = self.embed_network(x)
        out = self.linear_probe(embeds) 
        return out, embeds
    
    
    
class SigmoidLogisticRegression(nn.Module):
    def __init__(self, num_classes, shape=784):
        super(SigmoidLogisticRegression, self).__init__()

        self.num_classes = num_classes 
        if num_classes == 2: 
            self.fc = nn.Linear(shape, 1)
        else: 
            self.fc = nn.Linear(shape, num_classes)
        self.shape = shape 


    def forward(self, x):
        x = x.view(-1, self.shape)
        x = self.fc(x)
        return x


class SoftmaxLogisticRegression(nn.Module):
    def __init__(self, num_classes, shape=784):
        super(SoftmaxLogisticRegression, self).__init__()
        self.fc = nn.Linear(shape, num_classes) 
        self.shape = shape

    def forward(self, x):
        x = x.view(-1, self.shape)
        x = self.fc(x)
        return x

    
    
class ConvNetWithEmbeddingsEarly(nn.Module):
    def __init__(self, num_classes):
        super(ConvNetWithEmbeddingsEarly, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 10, kernel_size=5)
        self.conv1_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(250, 50)
        if (num_classes == 2): 
            self.fc2 = nn.Linear(50, 1) 
        else:
            self.fc2 = nn.Linear(50, num_classes)


    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1_drop(self.conv1(x)), 2))
        embed = F.max_pool2d((self.conv2(x)), 2)
        x = F.relu(embed)
        x = x.view(-1, 250) 
        x = self.fc1(x)
        x = F.relu(F.dropout(x, training=self.training))
        x = self.fc2(x)
        
        embed = embed.view(-1, 250) 
        return x, embed 
    
class ConvNetOnlyEmbeddingsEarly(nn.Module):
    def __init__(self, num_classes):
        super(ConvNetOnlyEmbeddingsEarly, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 10, kernel_size=5)
        self.conv1_drop = nn.Dropout2d()
        

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1_drop(self.conv1(x)), 2))
        embed = F.max_pool2d((self.conv2(x)), 2)
        embed = embed.view(-1, 250) 
        return embed

class ConvNetLinearProbeEarly(nn.Module): 
    def __init__(self, num_classes):
        super(ConvNetLinearProbeEarly, self).__init__()
        self.fc1 = nn.Linear(250, 50)
        if (num_classes == 2): 
            self.fc2 = nn.Linear(50, 1) 
        else:
            self.fc2 = nn.Linear(50, num_classes)
            
    def forward(self, embed):
        x = F.relu(embed)
        x = self.fc1(x)
        x = F.relu(F.dropout(x, training=self.training))
        x = self.fc2(x)
        return x
    
class ConvNetWithEmbeddingsEarly2(nn.Module):
    def __init__(self, num_classes):
        super(ConvNetWithEmbeddingsEarly2, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 10, kernel_size=5)
        self.conv1_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(250, 50)
        if (num_classes == 2): 
            self.fc2 = nn.Linear(50, 1) 
        else:
            self.fc2 = nn.Linear(50, num_classes)


    def forward(self, x):
        embed = self.conv1(x)
        x = F.relu(F.max_pool2d(self.conv1_drop(embed), 2))
        x = F.max_pool2d((self.conv2(x)), 2)
        x = F.relu(x)
        x = x.view(-1, 250) 
        x = self.fc1(x)
        x = F.relu(F.dropout(x, training=self.training))
        x = self.fc2(x)
        
        embed = torch.flatten(embed) 
        return x, embed 
    
class ConvNetOnlyEmbeddingsEarly2(nn.Module):
    def __init__(self, num_classes):
        super(ConvNetOnlyEmbeddingsEarly2, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)

    def forward(self, x):
        embed = self.conv1(x)
        return embed

class ConvNetLinearProbeEarly2(nn.Module): 
    def __init__(self, num_classes):
        super(ConvNetLinearProbeEarly2, self).__init__()
        self.conv2 = nn.Conv2d(10, 10, kernel_size=5)
        self.conv1_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(250, 50)
        if (num_classes == 2): 
            self.fc2 = nn.Linear(50, 1) 
        else:
            self.fc2 = nn.Linear(50, num_classes)
            
    def forward(self, embed):
        x = F.relu(F.max_pool2d(self.conv1_drop(embed), 2))
        x = F.max_pool2d((self.conv2(x)), 2)
        x = F.relu(x)
        x = x.view(-1, 250) 
        x = self.fc1(x)
        x = F.relu(F.dropout(x, training=self.training))
        x = self.fc2(x)
        return x
    