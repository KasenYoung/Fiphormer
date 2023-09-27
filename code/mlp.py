import os
import torch
import torch_geometric.nn as pyg
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from dataloader_singlegraph import  GraphDataset_SG
from torch.utils.data import DataLoader
from torch_geometric.utils import to_networkx
import numpy as np
import networkx as nx


class MLP(nn.Module):
    def __init__(self) :
        super().__init__()
        self.l1=nn.Linear(26,100)
        self.l2=nn.Linear(100,26)
        self.l3=nn.Linear(26,1)
        self.activation=nn.ReLU()
    
    def forward(self,x):
        #x=x.view(-1,18*26)
        x=self.l1(x)
        x=self.activation(x)
        x=self.l2(x)
        x=self.activation(x)
        x=self.l3(x)
        return x
    

def training(trainloader,  model, loss_fn, optimizer, epoches):
    # h_t=torch.zeros(batch_size,N,hidden_dim)
    model.train()
    num=0
    #batch_size=0
    for epoch in range(epoches):
        loss_train= torch.tensor(0.0)
        for batch in trainloader:

            feats,label=batch[0].float(),batch[3].float()
            #print(feats,label)
            num+=feats.shape[0]
            # print(loss)
            out=model(feats)
            loss=loss_fn(out,label)
            optimizer.zero_grad()
            #torch.autograd.set_detect_anomaly(True)
            loss.backward()

            optimizer.step()
            loss_train += loss.item()
        #if epoch==1 or epoch%10==0:
        print('Training:The {} th epoch loss is: {},the average loss for each graph is {}.'.format(epoch, loss_train,loss_train/num))
        '''
            for name, parms in model.named_parameters():
                print('-->name:', name, '-->grad_requirs:', parms.requires_grad,
                      ' -->grad_value:', parms.grad)
        '''

def test(testloader,  model, loss_fn):
    # h_t=torch.zeros(batch_size,N,hidden_dim)
    model.eval()
    num=0
    
    loss_test= torch.tensor(0.0)
    for batch in testloader:

        feats,label=batch[0].float(),batch[3].float()
        
        num+=feats.shape[0]
        # print(loss)
        out=model(feats)
        loss=loss_fn(out,label)
        
        loss_test += loss.item()
        #if epoch==1 or epoch%10==0:
    print('Testing: The total loss is: {},the average loss for each graph is {}.'.format(loss_test,loss_test/num))
    '''
            for name, parms in model.named_parameters():
                print('-->name:', name, '-->grad_requirs:', parms.requires_grad,
                      ' -->grad_value:', parms.grad)
        '''

if __name__ == '__main__':
    os.chdir('graphs')
    
    loss_fn = nn.MSELoss()
    trainset = GraphDataset_SG(path='./',train=True,test_rate=0.2)
    testset = GraphDataset_SG(path='./', train=False,test_rate=0.2)
    trainloader = DataLoader(dataset=trainset, batch_size=4, shuffle=True)
    testloader = DataLoader(dataset=testset, batch_size=4, shuffle=True)
    model = MLP()
    optimizer = optim.AdamW(model.parameters(), lr=0.01)
    training(trainloader=trainloader,  model=model,loss_fn=loss_fn,  optimizer=optimizer, epoches=20)
    test(testloader=testloader,  model=model, loss_fn=loss_fn)