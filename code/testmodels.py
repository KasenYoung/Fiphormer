import os
import torch
import torch_geometric.nn as pyg
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from dataloader import  GraphDataset
from torch.utils.data import DataLoader
from torch_geometric.utils import to_networkx
import numpy as np
import networkx as nx


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, alpha=2, dropout=0.2, concat=True):
        super().__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, x, adj):
        Wh = torch.matmul(x, self.W)
        a_input = self.prepare_attention_input(Wh)
        # print(a_input.shape)
        e = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(3)
        # print(e.shape)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        # print(attention.shape)
        attention = F.softmax(attention, dim=2)
        # print(attention.shape)
        attention = F.dropout(attention, self.dropout, training=self.training)

        h_prime = torch.matmul(attention, Wh)
        
        #print(h_prime.shape)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def prepare_attention_input(self, Wh):  # Wh:(b,N,out_features)
        N = Wh.size()[1]

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=1)
        Wh_repeated_alternating = Wh.repeat(1, N, 1)
        all_combine_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=2)

        return all_combine_matrix.view(-1, N, N, 2 * self.out_features)
    


class TemporalLayer(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim,act_func=nn.ReLU()) :
        super().__init__()
        self.hidden_dim=hidden_dim
        self.out_dim=output_dim
        self.act_func=act_func
        self.x_proj=nn.Linear(in_features=input_dim,out_features=hidden_dim,bias=True)
        self.h_proj=nn.Linear(in_features=hidden_dim,out_features=hidden_dim,bias=True)
        self.out_proj=nn.Linear(in_features=hidden_dim*2,out_features=output_dim,bias=True)


    def forward(self,x,x_hidden,h): #x:b*N*d,x_hidden and h:b*N*hidden_dim

        h_new=self.x_proj(x)+self.h_proj(h)
        total_feature = torch.cat((x_hidden, h_new), dim=2)
        output = self.act_func(self.out_proj(total_feature))
        return output,h_new


class LSTMLayer(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim,act_func=nn.ReLU()) :
        super().__init__()
        self.hidden_dim=hidden_dim
        self.out_dim=output_dim
        self.act_func=nn.Tanh()
        self.W_f=nn.Linear(in_features=hidden_dim,out_features=hidden_dim,bias=True)
        self.W_i = nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True)
        self.W_o = nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True)
        self.W_c = nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True)
        self.U_f=nn.Linear(in_features=input_dim,out_features=hidden_dim,bias=True)
        self.U_i=nn.Linear(in_features=input_dim,out_features=hidden_dim,bias=True)
        self.U_o=nn.Linear(in_features=input_dim,out_features=hidden_dim,bias=True)
        self.U_c = nn.Linear(in_features=input_dim, out_features=hidden_dim, bias=True)
        self.sigmoid=nn.Sigmoid()
        self.out_proj = nn.Linear(in_features=hidden_dim * 2, out_features=output_dim, bias=True)
        self.gate=act_func



    def forward(self,x,x_hidden,c,h): #x:b*N*d,x_hidden and c:b*N*hidden_dim h:b*N*hidden_dim
        f_t=self.sigmoid(self.W_f(h)+self.U_f(x))
        i_t=self.sigmoid(self.W_i(h)+self.U_i(x))
        o_t = self.sigmoid(self.W_o(h) + self.U_o(x))
        c_hat=self.act_func(self.W_c(h) + self.U_c(x))
        c_t=f_t*c+i_t*c_hat
        h_t=o_t*self.gate(c_t)
        total_feature=torch.cat((x_hidden,h_t),dim=2)
        output=self.gate(self.out_proj(total_feature))

        return output,h_t,c_t

class JustALayer(nn.Module):
    def __init__(self,hidden_dim,output_dim,act_func=nn.ReLU()) :
        super().__init__()
        self.hidden_dim=hidden_dim
        self.out_dim=output_dim
        self.act_func=act_func
        self.x_proj=nn.Linear(in_features=hidden_dim,out_features=hidden_dim,bias=True)
        self.h_proj=nn.Linear(in_features=hidden_dim,out_features=hidden_dim,bias=True)
        self.out_proj=nn.Linear(in_features=hidden_dim,out_features=output_dim,bias=True)


    def forward(self,x_hidden,h): #x:b*N*d,x_hidden and h:b*N*hidden_dim

        h_new=self.x_proj(x_hidden)+self.h_proj(h)

        output = self.act_func(self.out_proj(h_new))
        return output,h_new

class MLP(nn.Module):
    def __init__(self,in_features,out_features,dropout=0.2) :
        super().__init__()
        self.l1=nn.Linear(in_features=in_features,out_features=2*in_features,bias=True)
        self.l2=nn.Linear(in_features=2*in_features,out_features=out_features,bias=True)
        
        self.activation=nn.ReLU()
        self.dropout=nn.Dropout(p=dropout)
    def forward(self,x):
        #x=x.view(-1,18*26)
        x=self.l1(x)
        #print(x.shape)
        x=self.dropout(self.activation(x))
        x=self.l2(x)
        
        return x
    

class SimpleModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, alpha=2, dropout=0.2, act_func=nn.ReLU(), concat=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.alpha = alpha
        self.dropout = dropout
        self.act_func = act_func
        self.concat = concat
        self.graph_conv = GraphAttentionLayer(in_features=input_dim, out_features=hidden_dim, alpha=alpha,
                                              dropout=dropout, concat=self.concat)
        self.temporal_layer = TemporalLayer(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                                            act_func=self.act_func)

    def forward(self, x, adj, h):  # x:N*d, adj:N*N,h:N*h
        x_hidden = self.graph_conv(x, adj)
        #print(x_hidden.shape)
        output, h_new = self.temporal_layer(x, x_hidden, h)
        # print('reaches here!')

        return output, h_new


class SimpleLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, alpha=2, dropout=0.2, act_func=nn.ReLU(), concat=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.alpha = alpha
        self.dropout = dropout
        self.act_func = act_func
        self.concat = concat
        self.graph_conv = GraphAttentionLayer(in_features=input_dim, out_features=hidden_dim, alpha=alpha,
                                              dropout=dropout, concat=self.concat)
        self.temporal_layer = LSTMLayer(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                                            act_func=self.act_func)

    def forward(self, x, adj, c,h):  # x:N*d, adj:N*N,h:N*h
        x_hidden = self.graph_conv(x, adj)
        output, h_t,c_t = self.temporal_layer(x, x_hidden, c,h)
        # print('reaches here!')

        return output, h_t,c_t
    

class SimpleMLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, alpha=2, dropout=0.2, act_func=nn.ReLU(), concat=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.alpha = alpha
        self.dropout = dropout
        self.act_func = act_func
        self.concat = concat
        self.graph_conv = MLP(in_features=input_dim,out_features=hidden_dim,dropout=dropout)
        self.temporal_layer = TemporalLayer(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                                            act_func=self.act_func)

    def forward(self, x, adj, h):  # x:N*d, adj:N*N,h:N*h
        x_hidden = self.graph_conv(x)
        output, h_new = self.temporal_layer(x, x_hidden, h)
        # print('reaches here!')

        return output, h_new


def compute_loss(adj,model,loss_fn,X,Y,N,interval,hidden_dim,batch_size):
    loss=torch.tensor(0.0)
    h_t=torch.zeros(batch_size,N,hidden_dim)

    for x,y  in zip(torch.split(X,1,dim=1),torch.split(Y,1,dim=1)):
        x=x.squeeze(1)
        y=y.squeeze(1)
        out_t,h_t=model(x,adj,h_t)
        loss=loss+loss_fn(out_t,y)

    loss=loss/interval
    return loss

def compute_singleloss(adj,model,loss_fn,X,Y,N,interval,hidden_dim,batch_size):
    loss=torch.tensor(0.0)
    h_t=torch.zeros(batch_size,N,hidden_dim)
    cnt=0
    for x,y  in zip(torch.split(X,1,dim=1),torch.split(Y,1,dim=1)):
        x=x.squeeze(1)
        y=y.squeeze(1)
        out_t,h_t=model(x,adj,h_t)
        cnt+=1
        if cnt==interval:
            loss=loss_fn(out_t,y)

    #loss=loss/interval
    return loss

def compute_lstmloss(adj,model,loss_fn,X,Y,N,interval,hidden_dim,batch_size):
    loss=torch.tensor(0.0)
    h_t=torch.zeros(batch_size,N,hidden_dim)
    c_t = torch.zeros(batch_size, N, hidden_dim)

    for x,y  in zip(torch.split(X,1,dim=1),torch.split(Y,1,dim=1)):
        x=x.squeeze(1)
        y=y.squeeze(1)
        out_t,h_t,c_t=model(x,adj,c_t,h_t)
        loss=loss+loss_fn(out_t,y)

    loss=loss/interval
    return loss


def training(trainloader, adj, model, loss_fn, optimizer, epoches, N, interval, hidden_dim, str='rnn'):
    # h_t=torch.zeros(batch_size,N,hidden_dim)
    model.train()
    num=0
    batch_size=0
    for epoch in range(epoches):
        loss_train= torch.tensor(0.0)
        for batch in trainloader:
     
            X = batch[0].float()  # node_feature
            batch_size=X.shape[0]
            num+=batch_size
            Y = batch[3].float()  # label
            X=X.permute(1,0,2,3)
            norm=nn.BatchNorm2d(num_features=batch_size,eps=1e-5)
            X=norm(X)
            X = X.permute(1, 0, 2, 3)
            if str=='rnn':
                loss = compute_loss(adj, model, loss_fn, X, Y, N, interval, hidden_dim, batch_size)
            elif str=='lstm':
                loss = compute_lstmloss(adj, model, loss_fn, X, Y, N, interval, hidden_dim, batch_size)
          
            optimizer.zero_grad()
            #torch.autograd.set_detect_anomaly(True)
            loss.backward()

            optimizer.step()
            loss_train += loss.item()
        #if epoch==1 or epoch%10==0:
        print('The {} th epoch loss is: {},the average loss for each graph is {}.'.format(epoch, loss_train,loss_train/num))
        '''
            for name, parms in model.named_parameters():
                print('-->name:', name, '-->grad_requirs:', parms.requires_grad,
                      ' -->grad_value:', parms.grad)
        '''

def test(testloader, adj, model, loss_fn,  N, interval, hidden_dim,str='rnn'):
    # h_t=torch.zeros(batch_size,N,hidden_dim)
    model.eval()
    num=0
    batch_size=0
    loss_test= torch.tensor(0.0)
    for batch in testloader:

        X = batch[0].float()  # node_feature
        batch_size=X.shape[0]
        num+=batch_size
        Y = batch[3].float()  #
        X=X.permute(1,0,2,3)
        norm=nn.BatchNorm2d(num_features=batch_size,eps=1e-5)
        X=norm(X)
        X = X.permute(1, 0, 2, 3)
        
        if str=='rnn':
            loss = compute_singleloss(adj, model, loss_fn, X, Y, N, interval, hidden_dim, batch_size)
        elif str=='lstm':
            loss = compute_lstmloss(adj, model, loss_fn, X, Y, N, interval, hidden_dim, batch_size)
        # print(loss)

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
    text1 = os.listdir()[15]
    data1 = torch.load(text1)
    G = to_networkx(data=data1)
    adj = nx.adjacency_matrix(G)
    adj = np.array(adj.todense())
    adj = torch.from_numpy(adj).long()
    loss_fn = nn.MSELoss()
    trainset = GraphDataset(path='./', interval=5,train=True,test_rate=0.2)
    testset = GraphDataset(path='./', interval=5, train=False,test_rate=0.2)
    trainloader = DataLoader(dataset=trainset, batch_size=4, shuffle=True)
    testloader = DataLoader(dataset=testset, batch_size=4, shuffle=True)
    
    print('GAT:')
    model = SimpleModel(input_dim=26, hidden_dim=16, output_dim=1, alpha=2, dropout=0.2, act_func=nn.ReLU(),
                        concat=True)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    training(trainloader=trainloader, adj=adj, model=model, loss_fn=loss_fn, N=18, hidden_dim=16, 
             interval=5, optimizer=optimizer, epoches=20,str='rnn')
    test(testloader=testloader, adj=adj, model=model, loss_fn=loss_fn, N=18, hidden_dim=16,
             interval=5,  str='rnn')
    print('-----------------------------------------------------------------------')
    
    print("MLP:")
    model = SimpleMLPModel(input_dim=26, hidden_dim=16, output_dim=1, alpha=2, dropout=0.2, act_func=nn.ReLU(),
                        concat=True)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    training(trainloader=trainloader, adj=adj, model=model, loss_fn=loss_fn, N=18, hidden_dim=16, 
             interval=5, optimizer=optimizer, epoches=20,str='rnn')
    test(testloader=testloader, adj=adj, model=model, loss_fn=loss_fn, N=18, hidden_dim=16,
             interval=5,  str='rnn')
    
    