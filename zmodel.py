## environment
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np
from torch.utils.data import DataLoader
##  function

# generate current, previous and next set
def set_generate(x):  # input NxM signal
    set_all = []
    N, M = x.shape # M is the number of data points of each epoch, N is the number of epoch
    x_0 = x[0, :] # data of the first epoch 
    x_f = x[-1, :] # data of the last epoch
    x_current = x # current set
    set_all.append(x_current)
    x_previous = np.roll(x, 1, axis=0)
    x_previous[0, :] = x_0  # ''previous'' set
    set_all.append(x_previous)
    x_next = np.roll(x, -1, axis=0)
    x_next[0, :] = x_f # ''next'' set
    set_all.append(x_next)
    # combine 3 sets
    return set_all

# network CNN, which consists of 2 different CNNs: CNN1 and CNN2
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10 , (1, 55), padding='same') # CNN1
        self.pool = nn.MaxPool2d(kernel_size=(1, 16), ceil_mode=True)
        self.conv2 = nn.Conv2d(10, 5, (1, 25), padding='same') # CNN2
        self.batch1 = nn.BatchNorm2d(10)
        self.batch2 = nn.BatchNorm2d(5)
        self.dense1 = nn.Linear(60, 10) # dense
        self.dense2 = nn.Linear(10, 5) # dense
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.pool(self.batch1(F.relu(self.conv1(x)))) # CNN1
        x = self.pool(self.batch2(F.relu(self.conv2(x)))) # CNN2
        x = self.flatten(x)
        x = F.relu(self.dense1(x)) # relu activation
        x = F.softmax(self.dense2(x)) # softmax activation
        return x


# network BiLSTM, using the extracted features from CNN
class BiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.LSTM = nn.LSTM(hidden_size=128, bidirectional=True)
        self.dense = nn.Linear(256, 5)

    def forward(self, x):
        x = F.softmax(self.dense(self.LSTM(x)), dim=1)
        return x


## Training CNN
# parameter setting

model = CNNModel()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
x_train = np.random.rand(3,3000)
y_train = np.array(['W', 'N1', 'N2'])
trainloader = DataLoader((x_train, y_train))
submodel_name = np.array(['W', 'N1', 'N2', 'N3', 'REM'])
set_all = set_generate(x_train) # generate a set containing 3 sets
max_iter = 10 # maximum iteration
tol_CNN = 10 # tolerance for CNN training
features = [] # get the empty feature list


## Start
for set in set_all:
    loss = 0
    i = 0
    for stage in submodel_name:
        Nc = np.sum((y_train==stage))
        N = len(y_train)
        p1 = Nc/N # probability of 'stage' in the corresponding model
        p2 = 1-p1 # probability of other 'stage' in this model
        W = np.zeros(N)
        W[:] = p2
        ind_corres = np.where(y_train==stage)
        W[ind_corres] = p1
        w = 0.5 / W
        w = torch.tensor(w)
        loss_fn = nn.CrossEntropyLoss(weight = w) # using entropy loss
        loss_all = 0
        # training each sub-model
        for iter in range(max_iter):
            for inputs, labels in trainloader:
                y_pre = model(inputs)
                loss_sub = loss_fn(y_pre, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        loss_all = loss_all + loss_sub # compute the toal loss of 5 sub-models
    if loss_all < tol_CNN:
        continue
    features.append(model(set))
# The step above should generate N x 75 matrix where 75 means the number of features extracted

## Training BiLSTM
# parameters
model = BiLSTM()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
x_train = features
trainloader = DataLoader((x_train, y_train))
max_iter = 10 # maximum iteration
tol_LSTM = 10 # tolerance for CNN training
loss_fn = nn.CrossEntropyLoss()
# Start
for iter in range(iter):
    for inputs, labels in trainloader:
        y_pre = model(x_train)
        loss = loss_fn(y_pre, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if loss < tol_LSTM:
        break
outputs = y_pre
# The step above should genertate a N x 1 vector storing the predicted sleep stage (i.e., 'W', 'N1-N3', 'REM')
