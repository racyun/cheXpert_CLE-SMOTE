## Notebooks 
The main notebooks used to test are convnet_2_class.ipynb and convnet_3_class.ipynb. The notebook auc_graphs.ipynb is for graphing results. 

## Python Files

#### class_sampling.py
This file contains the code for resampling the datasets. 

```
reduced_train_CIFAR10 = class_sampling.Reduce(train_CIFAR10, NUM_CLASSES, transform=transform)
ratio_train_CIFAR10 = class_sampling.Ratio(train_CIFAR10, NUM_CLASSES, ratio, transform=transform)
```

#### models.py
This file contains the models to be used. Some models return embeddings. 

```
network = models.ConvNet(2) # pass in number of classes
```


#### train.py
This file contains code for training. 

```
for epoch in range(n_epochs):
  train_counter, train_losses = train.train_sigmoid(epoch, train_loader, network, optimizer, verbose=False)
```

#### loss_fns.py
This file contains custom coded loss functions. Can be passed into the train function. 

```
loss_fn = loss_fns.CappedBCELoss
loss_fn_args = {}
loss_fn_args['loss_cap'] = cap
for epoch in range(n_epochs):
  _,_ = train.train_sigmoid(epoch, train_loader, network, optimizer, verbose=False, loss_fn=loss_fn, loss_fn_args=loss_fn_args)

```

#### metric_utils.py
This file contains the code for evaluating the model on test data. 
```
_, auc = metric_utils.auc_sigmoid(test_loader_reduced, network)
```

#### inference.py
This file contains code for testing, to be used in metric_utils.py 
