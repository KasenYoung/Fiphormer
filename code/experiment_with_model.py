import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch.utils.data import DataLoader, Subset, random_split
import csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pdb
import os
from torch.nn.parameter import Parameter
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 遍历graph文件夹里的csv文件
graph_folder = 'graph'
csv_files = [os.path.join(graph_folder, file) for file in os.listdir(graph_folder) if file.endswith('.csv')]


# Function to read CSV file and move label column to the last
def read_csv_and_move_label(csv_file):
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        label_index = header.index("label")
        id_index = header.index("id")
        time_index = header.index("snapshot_ts")
        data = [[float(col) if i != id_index and i != time_index else col for i, col in enumerate(row)] for row in reader]
        # Move label column to the last
        for row in data:
            label = row.pop(label_index)
            row.append(label)
            del row[time_index]
            del row[id_index]

        # 对特征进行缩放
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)

    return scaled_data

# 构建图数据
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        node_features = torch.tensor(self.data[idx][:-1], dtype=torch.float)
        label = torch.tensor(self.data[idx][-1], dtype=torch.float)
        return node_features, label

class Aggr_layer(nn.Module):

    def __init__(self, in_features, out_features, alpha):
        super(Aggr_layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, input_emb):
        x = torch.matmul(adj, input)
        return (1 - self.alpha) * x + self.alpha * input_emb


class GATDropModel(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayers=12, dropout=0.3, alpha=0.2):
        super(GATDropModel, self).__init__()  # Corrected superclass call
        self.midlayers = nn.ModuleList()
        for _ in range(nlayers):
            self.midlayers.append(Aggr_layer(nhid, nhid, alpha))
        self.input_layer, self.output_layer = GATConv(nfeat, nhid, heads=nlayers), nn.Linear(nhid * nlayers, nclass)
        self.conv2 = GATConv(hidden_dim * nlayers, hidden_dim, heads=nlayers)
        self.alpha = alpha
        self.activation = F.relu
        self.dropout = dropout

    def forward(self, x):
        num_nodes = x.size(0)
        edge_index = torch.tensor([[i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j],
                                  dtype=torch.long).t()
        x = self.input_layer(x, edge_index)
        x, edge_weight = self.conv2(x, edge_index, return_attention_weights=True)
        chunked_tensors = torch.chunk(edge_weight[1], len(self.midlayers), dim=1)

        # 将分块后的张量存储在列表中
        adj = [chunk.view(100, 100) for chunk in chunked_tensors]
        collector = [x]
        for _ in range(len(self.midlayers)):
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.activation(self.midlayers[_](x, adj if not isinstance(adj, list) else adj[_], collector[0]))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.output_layer(x).squeeze(1)
        # x = F.log_softmax(x, dim=1)
        return x


class GATBase(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(GATBase, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=12)
        self.conv2 = GATConv(hidden_dim * 12, hidden_dim, heads=12)
        self.lin = nn.Linear(hidden_dim * 12, num_classes)

    def forward(self, x):
        num_nodes = x.size(0)
        edge_index = torch.tensor([[i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j],
                                  dtype=torch.long).t()
        x = self.conv1(x, edge_index)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.dropout(x, p=0.5, training=self.training)
        x=self.lin(x).squeeze(1)
        return x
    
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim*4)
        self.fc3 = nn.Linear(hidden_dim*4, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x).squeeze(1)

        return x


# 设置参数
input_dim = 217 # 输入特征的维度
hidden_dim = 32
num_classes = 1  # 标签类别数
learning_rate = 1e-5
epochs = 200

# 创建 DataLoader
train_loaders = []
val_loaders = []
test_loaders = []
for csv_file in csv_files:
    data = read_csv_and_move_label(csv_file)
    dataset = CustomDataset(data)
    train_size = int(0.6 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # 分batch
    batch_size = 100
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    train_loaders.append(train_loader)
    val_loaders.append(val_loader)
    test_loaders.append(test_loader)

# 构建模型和优化器
models = [GATDropModel(input_dim, hidden_dim, num_classes), GATBase(input_dim, hidden_dim, num_classes),
          MLP(input_dim, hidden_dim, num_classes)]

criterion = nn.MSELoss()

train = []
val = []

for model in models:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-7)

    # 训练模型
    train_losses = []
    val_losses = []
    test_losses = []
    best_val_loss = float('inf')
    patience = 5  # 设定容忍度，即连续多少个epoch没有验证集上的损失提升时停止训练
    early_stopping_counter = 0

    # 定义学习率调度器
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=patience, factor=0.1, verbose=True)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for train_loader in train_loaders:
            for node_features, labels in train_loader:  # 直接从data_loader中获取node_features和labels
                optimizer.zero_grad()

                out = model(node_features)  # 使用生成的边索引\
                # print(out.shape, labels.shape)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * node_features.size(0)

        train_losses.append(train_loss / (len(train_loaders[0].dataset) * len(train_loaders)))

        # 在验证集上计算损失
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for val_loader in val_loaders:
                for data, labels in val_loader:
                    # out = model(node_features, adj_matrix)  # 使用生成的边索引\
                    outputs = model(data)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * data.size(0)  # 累计验证损失
                    total += labels.size(0)

            val_losses.append(val_loss / (len(val_loaders[0].dataset) * len(val_loaders)))

        scheduler.step(val_loss)

        # 计算并打印平均损失和验证准确率
        avg_train_loss = train_loss / (len(train_loaders[0].dataset) * len(train_loaders))
        avg_val_loss = val_loss / (len(val_loaders[0].dataset) * len(val_loaders))
        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            early_stopping_counter = 0  # 重置早停计数器
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= patience:
            print(f'Validation loss hasn\'t improved for {patience} epochs. Early stopping...')
            break

    print('Finished Training, best_val_loss:', best_val_loss)
    # 在测试集上评估模型
    model.eval()
    with torch.no_grad():
        for test_loader in test_loaders:
            for x_batch, y_batch in test_loader:
                out = model(x_batch)
                mse = criterion(out, y_batch)
                test_losses.append(mse.item())
    print('Test Loss:', sum(test_losses) / len(test_losses))
    print('---------------------------------------------------')
    train.append(train_losses)
    val.append(val_losses)

# 画出损失下降图
plt.figure()
plt.plot(range(len(train[0])), train[0], label='GATDropModel')
plt.plot(range(len(train[1])), train[1], label='GAT')
plt.plot(range(len(train[2])), train[2], label='MLP')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.savefig('train_loss.png')
plt.show()

plt.figure()
plt.plot(range(len(val[0])), val[0], label='GATDropModel')
plt.plot(range(len(val[1])), val[1], label='GAT')
plt.plot(range(len(val[2])), val[2], label='MLP')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation Loss')
plt.legend()
plt.savefig('val_loss.png')
plt.show()
