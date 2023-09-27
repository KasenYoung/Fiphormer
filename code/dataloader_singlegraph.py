#data loader created by KasenYoung,2023-08-10.
import os
import numpy as np
import torch 
import torch_geometric as pyg
from torch.utils.data import DataLoader,Dataset

#define dataset
class GraphDataset_SG(Dataset):
    def __init__(self,path,test_rate=0.2,train=True):
        '''
        path:the path where the 'pt' files are stored.
        interval:time interval.
        '''
        super().__init__()
        self.path=path
        self.test_rate=test_rate
        
        self.train=train
        
        file_ls=[]
        for file in os.listdir(self.path):
            file_ls.append(file)
        
        
        self.nums=len(file_ls)
        self.test_num=int(self.nums*self.test_rate)
        self.train_num=self.nums-self.test_num

        if self.train:
            sample_ls=file_ls[0:self.train_num] #drop the first some files.
        else:
            sample_ls=file_ls[self.train_num:]
        self.sample_ls=sample_ls
        


    def __len__(self):
        if self.train:
            return self.train_num
        else:
            return  self.test_num
    def __getitem__(self,index):
        #self.data_ls=self.partition(self.sample_ls,self.interval) #partition the sample_ls
        begin_pos=self.sample_ls[index] # the current list of file names

        #initialize
       
        try:
            data_begin=torch.load(begin_pos)
        except:
            os.chdir(self.path)
            data_begin=torch.load(begin_pos)

        x_pack,edge_index_pack,edge_attr_pack,label_pack=data_begin.x,data_begin.edge_index,data_begin.edge_attr,data_begin.y
        #x_shape,edge_index_shape,edge_attr_shape,label_shape=x_pack.shape,edge_index_pack.shape,edge_attr_pack.shape,label_pack.shape #determine shape

        

      

        return x_pack,edge_index_pack,edge_attr_pack,label_pack
        

#test
if __name__ == '__main__':
    #you can try to use GraphDataset here.
    dataset=GraphDataset_SG(path='graphs')  #path:can be absolute path or relative path.Make sure this file is in the same directory as the 'graphs' directory.
    dataloader=DataLoader(dataset=dataset,batch_size=2,shuffle=True)
    print(len(dataloader)) #total files/(batch_size*interval)
    for data in dataloader:
        print(data[0].shape,data[1].shape,data[2].shape,data[3].shape)