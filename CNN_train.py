# Function for training CNN
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from set_generate import set_generate
from torch.utils.data import DataLoader
import numpy as np



# epochs_train, events_train should be numpy array

def CNN_train(epochs_train, events_train):

    x_current, x_previous, x_next = set_generate(epochs_train) # outputs are numpy array

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
            self.relu = nn.ReLU()
            self.softmax = nn.Softmax(dim=1)

        def forward(self, x):
            x = self.pool(self.batch1(self.relu(self.conv1(x)))) # CNN1
            x = self.pool(self.batch2(self.relu(self.conv2(x)))) # CNN2
            x = self.flatten(x)
            x = self.relu(self.dense1(x)) # relu activation
            x = self.softmax((self.dense2(x))) # softmax activation
            return x
        

    class MyLoss(nn.Module):
        def __init__(self, weight):
            super(MyLoss, self).__init__()
            self.weight = weight
        
        def forward(self, predictions, target): # prediction size: Nx5; target size: Nx1
            N, c = predictions.shape
            indx_row = torch.Tensor(np.arange(N)).int()
            indx_col = target.int()
            indx_predictions = (indx_row, indx_col)
            w = self.weight
            predictions = predictions[indx_predictions]
            loss = -torch.sum(torch.log(w * predictions))
            loss = loss / N
            #loss = 0
            #for i in range(5): # class i
            #   indx_row = torch.nonzero(target.int() == i)
            #   indx_col = i
            #   w = self.weight
            #   w = w[indx_row]
            #   subloss = -torch.sum(torch.log(w * predictions[indx_row, indx_col]))
            #   loss = loss + subloss
            #loss = loss / len(target)
            return loss
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    max_iter = 300
    features = []
    tol_CNN = 0.01
    
    filepath = ['current-0.pt', 'current-1.pt', 'current-2.pt', 'current-3.pt', 'current-4.pt', 
                'previous-0.pt', 'previous-1.pt', 'previous-2.pt', 'previous-3.pt', 'previous-4.pt'
                'next-0.pt', 'next-1.pt', 'next-2.pt', 'next-3.pt', 'next-4.pt']
    j = 0
    for set in (x_current, x_previous, x_next):
        for i in range(5): # i indicates the index of 5 sub-model, ranging from 0 to 4
            Nc = np.sum(events_train == i)
            N = len(events_train)
            p1 = Nc/N
            p2 = 1 - p1
            w = np.zeros(N)
            w[:] = p2
            ind_corres = np.where(events_train==i)
            w[ind_corres] = p1
            w = 0.5 / w
             # get the weight for each epoch
            model = CNNModel()
            model.to(device)
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
            trainloader = DataLoader(list(zip(set, events_train, w)), batch_size=64)
            for iter in range(max_iter):
                loss_all = 0
                for inputs, labels, w in trainloader:
                    inputs, labels, w = inputs.to(device), labels.to(device), w.to(device)
                    loss_fn = MyLoss(weight = w)
                    y_pre = model(inputs)
                    loss = loss_fn(y_pre, labels)
                    loss_all = loss + loss_all
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                print('iteration:', iter, 'loss:', loss_all)
            torch.save(model.state_dict(), filepath[j])
            j = j + 1
    print('finished CNN training')









