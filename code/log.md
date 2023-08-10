# 数据初步处理:graphs文件夹 
Updated:KasenYoung,2023-08-09.
将HenryZhou的txt文件解析并封装成pt文件，每个txt生成一张图。
data数据说明如下:
每个data为一个pyg.Data类型对象，代表一张全连接图。
data.x:结点特征矩阵，维度为18*26.
data.edge_index:邻接表矩阵，全连接方式，注意：无向图需要添加反向的边，故维度为2*306.
data.edge_attr:边的特征矩阵，维度为306*5.其中，edge_attr的每一行对应于edge_index的每一列所表示的边的特征。
data.y:结点特征矩阵，维度为18*1.
### 检查方式:
```Python
import os
import torch
from  torch_geometric.data import Data 

def loading():
    
    for i in os.listdir('graphs'):
        
        data=torch.load('graphs/'+i)
        '''
        #添加你进行的检查代码
        print(data)
        print(data.x)
        print(data.edge_index,data.edge_index.dtype)
        print(data.edge_attr,type(data.edge_attr))
        print(data.y)
        '''
    
loading()

```
# Dataset的创建
Updated:KasenYoung,2023-08-10.
仅仅根据前面生成的graph文件夹下的文件进行创建。目前共有301个文件。
基本思路是:根据时间间隔(interval)划分，共形成301//interval个对象，其中每个对象的构建是对应于文件列表中的一组文件生成。
GraphDataset的__getitem__函数每次返回一个元组，包含四个对象，依次是打包的结点特征、邻接矩阵、边特征、标签。
这四个张量均为四个维度，第一维为batch_size,第二维为设定的interval,其余两维为矩阵的本身维度。测试函数可写在main函数中。path可为绝对路径或相对路径，最好把此文件放在与'graph'文件夹同一级目录下。