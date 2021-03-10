# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 09:51:35 2020

@author: User
"""
import pandas as pd
import numpy as np

import torch
from torch import nn,optim
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader,TensorDataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import datetime
#from sklearn.preprocessing import OneHotEncoder
#DNN切訓練and測試資料
#encode area
clusters=pd.get_dummies(clusters, columns=['county'])
#data_str_ohe=onehotencoder.fit_transform(data_le).toarray()

X_train=clusters[clusters['num_nday']>0 & clusters['date']<datetime.datetime(2020,6,1)].drop(columns=['num_nday','date', 'inv_id','area',    'group_id',   'next_date',    'pre_date',
         'num_ndate',   'num_pdate', 'RMF_F']) 
X_test=clusters[clusters['num_nday']>0 & clusters['date']>=datetime.datetime(2020,6,1)].drop(columns=['num_nday','date', 'inv_id','area',    'group_id',   'next_date',    'pre_date',
         'num_ndate',   'num_pdate', 'RMF_F'])
y_train=clusters[clusters['num_nday']>0 & clusters['date']<datetime.datetime(2020,6,1)]['num_nday']
y_test=clusters[clusters['num_nday']>0 & clusters['date']>=datetime.datetime(2020,6,1)]['num_nday']

y_train = torch.tensor(y_train.values.astype(np.float32))
X_train = torch.tensor(X_train.values.astype(np.float32)) 
train_tensor = TensorDataset(X_train, y_train) 
train_loader = DataLoader(dataset = train_tensor, batch_size = 500, shuffle = True)

use_cuda = torch.cuda.is_available 
device = torch.device('cuda' if use_cuda else 'cpu')
torch.manual_seed()

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer
    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x
    
net = Net(n_feature=1, n_hidden=18, n_output=1).to(device) #you can use different n_hidden & lr for test
print(net)
optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
loss_func = torch.nn.MSELoss()
plt.ion()


for t in range(100):
    for i ,(X_train, y_train) in enumerate(train_loader):
        X_train=X_train.to(device)
        y_train=y_train.to(device)
        prediction = net(X_train)
        
        loss = loss_func(prediction, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if t % 10 == 0:
            plt.cla()
            plt.scatter(X_train.data.numpy(), y_train.data.numpy())
            plt.plot(X_train.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
            plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
            plt.show()
            plt.pause(0.1)
            plt.ioff()



#下面沒用到
class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 500, 1, 100, 1


# Construct our model by instantiating the class defined above
model = TwoLayerNet(D_in, H, D_out)

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = nn.MSELoss(reduction='sum')
optimizer = optim.SGD(model.parameters(), lr=1e-4)
for t in range(500):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(X_train)

    # Compute and print loss
    loss = criterion(y_pred, y_train)
    if t % 100 == 99:
        print(t, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.nn1 = nn.Linear(1, 15) #第一層 Linear NN
        self.nn2 = nn.Linear(15, 1) #第二層 Linear NN

    def forward(self, x):
        x = F.relu(self.nn1(x))  #對第一層 NN 使用Relu激活
        x = self.nn2(x)          #第二層直接輸出
        return x
model = Model()
print(model) #將模型print出來看看