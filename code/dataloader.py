#data loader created by KasenYoung,2023-08-10.
import os
import numpy as np
import torch 
import torch_geometric as pyg
from torch.utils.data import DataLoader,Dataset

#define dataset
class GraphDataset(Dataset):
    def __init__(self,path,interval,test_rate=0.2,train=True):
        '''
        path:the path where the 'pt' files are stored.
        interval:time interval.
        '''
        super().__init__()
        self.path=path
        self.test_rate=test_rate
        if not interval:
            raise RuntimeError('Given interval cannot be equal to 0!')
        self.train=train
        self.interval=interval
        file_ls=[]
        for file in os.listdir(self.path):
            file_ls.append(file)
        if len(file_ls) % interval !=0:
            print('The number of graphs is not divisible by interval.Could you have more appropiate intervals? ')
        self.left=len(file_ls)%self.interval
        self.nums=len(file_ls)//self.interval
        self.test_num=int(self.nums*self.test_rate)
        self.train_num=self.nums-self.test_num

        if self.train:
            sample_ls=file_ls[self.left:self.left+self.train_num*interval] #drop the first some files.
        else:
            sample_ls=file_ls[self.left+(self.train_num)*interval:]
        self.sample_ls=sample_ls
        self.data_ls = self.partition()



    def partition(self):

        return [self.sample_ls[i:i+self.interval] for i in range(0,len(self.sample_ls),self.interval)]

    def __len__(self):
        if self.train:
            return self.train_num
        else:
            return  self.test_num
    def __getitem__(self,index):
        #self.data_ls=self.partition(self.sample_ls,self.interval) #partition the sample_ls
        curr_pack=self.data_ls[index] # the current list of file names

        #initialize
        begin_pos=curr_pack[0]
        try:
            data_begin=torch.load(begin_pos)
        except:
            os.chdir(self.path)
            data_begin=torch.load(begin_pos)

        x_pack,edge_index_pack,edge_attr_pack,label_pack=data_begin.x,data_begin.edge_index,data_begin.edge_attr,data_begin.y
        x_shape,edge_index_shape,edge_attr_shape,label_shape=x_pack.shape,edge_index_pack.shape,edge_attr_pack.shape,label_pack.shape #determine shape

        #loop
        for i in range(1,len(curr_pack)):
            data_curr=torch.load(curr_pack[i])
            x_curr,edge_index_curr,edge_attr_curr,label_curr=data_curr.x,data_curr.edge_index,data_curr.edge_attr,data_curr.y
            x_pack=torch.cat((x_pack,x_curr),dim=0)
            edge_index_pack=torch.cat((edge_index_pack,edge_index_curr),dim=0)
            edge_attr_pack=torch.cat((edge_attr_pack,edge_attr_curr),dim=0)
            label_pack=torch.cat((label_pack,label_curr),dim=0)

        #reshape the tensors
        x_pack=x_pack.view(-1,x_shape[0],x_shape[1])
        edge_index_pack=edge_index_pack.view(-1,edge_index_shape[0],edge_index_shape[1])
        edge_attr_pack=edge_attr_pack.view(-1,edge_attr_shape[0],edge_attr_shape[1])
        label_pack=label_pack.view(-1,label_shape[0],label_shape[1])

        return x_pack,edge_index_pack,edge_attr_pack,label_pack
        

#test
if __name__ == '__main__':
    #you can try to use GraphDataset here.
    dataset=GraphDataset(path='graphs',interval=5)  #path:can be absolute path or relative path.Make sure this file is in the same directory as the 'graphs' directory.
    dataloader=DataLoader(dataset=dataset,batch_size=2,shuffle=True)
    print(len(dataloader)) #total files/(batch_size*interval)
    for data in dataloader:
        print(data[0].shape,data[1].shape,data[2].shape,data[3].shape)