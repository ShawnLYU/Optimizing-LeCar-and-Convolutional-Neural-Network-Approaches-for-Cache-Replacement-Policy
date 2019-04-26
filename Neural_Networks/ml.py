import copy
import torch
from torch import nn
import os
import numpy as np

from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('e', type=int) # epochs
parser.add_argument('p', type=str) # path


CACHE_SIZE=100
LR = 0.001  
BATCH_SIZE=32


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, CACHE_SIZE, 2)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=8,            # n_filters
                kernel_size=(30,3),              # filter size
                stride=1,                   # filter movement/step
                padding=0,                  # if want same width and length of this image afte#
            ),                              # output shape (16, 28, 28)
            nn.ReLU(),                      # activation
        )
        self.conv2 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv2d(
                in_channels=8,              # input height
                out_channels=16,            # n_filters
                kernel_size=(30,1),              # filter size
                stride=1,                   # filter movement/step
                padding=2),    # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
        )
        self.out = nn.Sequential(
            nn.Linear(3680,100),
            )
    def forward(self, x):
        x = self.conv1(x.unsqueeze(1))
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 *#
        output = self.out(x)
        return output




args = parser.parse_args()


EPOCH = args.e




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



model = CNN().to(device)



optimizer = torch.optim.Adam(model.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()   

the_epoch, pre_acc, best_model = 0, 0, None # keep tracking the one giving best validation accuacy

path = os.path.abspath(args.p)







################################### for batching



X_train = np.genfromtxt('./opt2/X_train.csv')
Y_train = np.genfromtxt('./opt2/Y_train.csv')
X_test = np.genfromtxt('./opt2/X_test.csv')
Y_test = np.genfromtxt('./opt2/Y_test.csv')



class LoadDotaDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __getitem__(self, item):
        feature = self.x[item]
        feature = feature.reshape(CACHE_SIZE,3)
        label = self.y[item]
        # label = np.zeros(CACHE_SIZE)
        # label[int(self.y[item])] = 1
        return feature, label
    def __len__(self):
        return len(self.x)




trainset = LoadDotaDataset(x=X_train, y=Y_train)
validset = LoadDotaDataset(x=X_test, y=Y_test)


train_loader = DataLoader(dataset=trainset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=validset, batch_size=BATCH_SIZE, shuffle=False)


# for train_data, train_label in train_loader:
#     print(train_data.shape)
###################################



def train(model, loss, optimizer, x, y, device):
    x, y = x.to(device), y.to(device).long()
    fx = model(x)
    output = loss(fx, y)
    optimizer.zero_grad()
    output.backward()
    optimizer.step()
    # return computeAccuracy(fx, y)
    return computeLoss(fx, y)


def test(model, loss, x, y, device):
    x, y = x.to(device), y.to(device).long()
    fx = model(x)
    # return computeAccuracy(fx, y)
    return computeLoss(fx, y)

 
def computeAccuracy(fx, y):
    prediction = torch.max(F.softmax(fx,dim=1), dim=1)[1]
    # values, indices = torch.max(F.softmax(fx), 0)
    pred_y = prediction.data.cpu().numpy()
    target_y = y.data.cpu().numpy()
    accuracy = sum(pred_y == target_y)/len(pred_y)
    return accuracy

def computeLoss(fx, y):
    _loss = torch.nn.CrossEntropyLoss()
    l = _loss(fx, y).item()
    return l



with open(os.path.join(path,'batch.log'),'a') as f:
    for i in range(EPOCH):
        train_acc, valid_acc = 0, 0
        pre_acc = 0
        for train_data, train_label in train_loader:
            train_acc = train(model, loss_func, optimizer, train_data.float(), train_label, device)
        for valid_data, valid_label in valid_loader:
            valid_acc = test(model, loss_func, valid_data.float(), valid_label, device)
        if valid_acc > pre_acc:
            best_model = copy.deepcopy(model)
            pre_acc = valid_acc
            the_epoch = i
        f.write('Epoch: %d| train loss: %.4f| test accuracy: %.2f | the best epoch: %d \n' % (i, train_acc, valid_acc, the_epoch))
        f.flush()
    model_path = os.path.join(path,'CNN.pth')
    torch.save(best_model.state_dict(), model_path)        
    f.write('Finished. Saving the model to %s\n' % model_path)
    f.write('The best validation accuracy occurs at %dth epoch' % (the_epoch))


# for train_data, train_label in train_loader:
#     x = train_data.float()
#     y = train_label
#     break
