import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

def LSTM_train(features, labels):

    class BiLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(75, 128, bidirectional=True)
            self.dense = nn.Linear(256, 5)
            self.softmax = nn.Softmax(dim = 0)

        def forward(self, x):
            x, _ = self.lstm(x)
            x = self.dense(x)
            x = self.softmax(x)
            return x
    
    class MyLoss(nn.Module):
        def __init__(self):
            super(MyLoss, self).__init__()

        def forward(self, predictions, labels):
            predictions = torch.argmax(predictions, dim=1)
            Na = torch.sum(predictions==labels)
            N = len(predictions)
            acc = Na / N
            return acc  

    
    train_loader = DataLoader(list(zip(features, labels)), batch_size = 64)
    max_iter = 3000
    loss_fn = MyLoss()
    model = BiLSTM()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    model.to(device)
    for iter in range(max_iter):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            y_pre = model(inputs)
            loss = loss_fn(y_pre, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Accuracy:', loss)
    torch.save(model.state_dict(), 'LSTM.pt')
    print('finished LSTM training')

