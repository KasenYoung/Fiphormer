import math
import torch
import torch.nn as nn

def init_params(module, n_layers):
    """
    初始化模型参数。
    对于线性层，使用正态分布进行参数初始化。
    对于嵌入层，使用正态分布进行参数初始化。
    """
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)

class GraphNodeFeature(nn.Module):
    """
    计算图中每个节点的特征。
    """

    def __init__(
        self, num_heads, num_atoms, num_in_degree, num_out_degree, hidden_dim, n_layers
    ):
        super(GraphNodeFeature, self).__init__()
        self.num_heads = num_heads
        self.num_atoms = num_atoms

        # 为结点特征构建嵌入层，+1表示一个额外的结点
        self.atom_encoder = nn.Embedding(num_atoms + 1, hidden_dim, padding_idx=0)
        self.in_degree_encoder = nn.Embedding(num_in_degree, hidden_dim, padding_idx=0) #入度的特征
        self.out_degree_encoder = nn.Embedding( 
            num_out_degree, hidden_dim, padding_idx=0
        )  #出度特征

        # 图的嵌入层
        self.graph_token = nn.Embedding(1, hidden_dim)

        # 初始化参数
        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, batched_data):
        # 获取输入的节点特征、入度和出度
        x, in_degree, out_degree = (
            batched_data["x"],
            batched_data["in_degree"],
            batched_data["out_degree"],
        )
        n_graph, n_node = x.size()[:2]

        # 节点特征 
        node_feature = self.atom_encoder(x).sum(dim=-2)  # [n_graph, n_node, n_hidden]

        # 将入度、出度特征加到节点特征上
        node_feature = (
            node_feature
            + self.in_degree_encoder(in_degree)
            + self.out_degree_encoder(out_degree)
        )

        # 获取图特征
        graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)

        # 合并图特征和节点特征得到最终的图节点特征
        graph_node_feature = torch.cat([graph_token_feature, node_feature], dim=1)

        return graph_node_feature


class GraphAttnBias(nn.Module):
    """
    计算每个头部的注意力偏置。
    """

    def __init__(
        self,
        num_heads,
        num_atoms,
        num_edges,
        num_spatial,
        num_edge_dis,
        hidden_dim,
        edge_type,
        multi_hop_max_dist,
        n_layers,
    ):
        super(GraphAttnBias, self).__init__()
        self.num_heads = num_heads
        self.multi_hop_max_dist = multi_hop_max_dist

        # 为边特征构建嵌入层，+1表示一个额外的边
        self.edge_encoder = nn.Embedding(num_edges + 1, num_heads, padding_idx=0)
        self.edge_type = edge_type
        if self.edge_type == "multi_hop":
            # 多跳边特征
            self.edge_dis_encoder = nn.Embedding(
                num_edge_dis * num_heads * num_heads, 1
            )
        self.spatial_pos_encoder = nn.Embedding(num_spatial, num_heads, padding_idx=0)

        # 图标记的虚拟距离嵌入层
        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)

        # 初始化参数
        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, batched_data):
        # 获取输入的注意力偏置、空间位置和节点特征
        attn_bias, spatial_pos, x = (
            batched_data["attn_bias"],
            batched_data["spatial_pos"],
            batched_data["x"],
        )
        edge_input, attn_edge_type = (
            batched_data["edge_input"],
            batched_data["attn_edge_type"],
        )

        n_graph, n_node = x.size()[:2]
        graph_attn_bias = attn_bias.clone()
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(
            1, self.num_heads, 1, 1
        )  # [n_graph, n_head, n_node+1, n_node+1]

        # 空间位置偏置
        # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
        spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).permute(0, 3, 1, 2)
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + spatial_pos_bias

        # 重置图标记的虚拟距离
        t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
        graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
        graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t

        # 边特征
        if self.edge_type == "multi_hop":
            # 多跳边特征
            spatial_pos_ = spatial_pos.clone()
            spatial_pos_[spatial_pos_ == 0] = 1  # 将填充位置设置为1
            # 将1保持为1，将x>1的位置设置为x - 1
            spatial_pos_ = torch.where(spatial_pos_ > 1, spatial_pos_ - 1, spatial_pos_)
            if self.multi_hop_max_dist > 0:
                spatial_pos_ = spatial_pos_.clamp(0, self.multi_hop_max_dist)
                edge_input = edge_input[:, :, :, : self.multi_hop_max_dist, :]
            # [n_graph, n_node, n_node, max_dist, n_head]
            edge_input = self.edge_encoder(edge_input).mean(-2)
            max_dist = edge_input.size(-2)
            edge_input_flat = edge_input.permute(3, 0, 1, 2, 4).reshape(
                max_dist, -1, self.num_heads
            )
            edge_input_flat = torch.bmm(
                edge_input_flat,
                self.edge_dis_encoder.weight.reshape(
                    -1, self.num_heads, self.num_heads
                )[:max_dist, :, :],
            )
            edge_input = edge_input_flat.reshape(
                max_dist, n_graph, n_node, n_node, self.num_heads
            ).permute(1, 2, 3, 0, 4)
            edge_input = (
                edge_input.sum(-2) / (spatial_pos_.float().unsqueeze(-1))
            ).permute(0, 3, 1, 2)
        else:
            # 普通边特征
            # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
            edge_input = self.edge_encoder(attn_edge_type).mean(-2).permute(0, 3, 1, 2)

        # 将边特征加到注意力偏置上
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + edge_input
        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)  # 重置

        return graph_attn_bias
