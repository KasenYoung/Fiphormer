# 导入必要的模块和库，其中fairseq是一个机器翻译库
from typing import Callable, Optional
import torch
import torch.nn as nn
from fairseq import utils
from fairseq.modules import LayerNorm #层归一化
from fairseq.modules.fairseq_dropout import FairseqDropout #dropout方法
from fairseq.modules.quant_noise import quant_noise #添加噪声??
from .multihead_attention import MultiheadAttention

class GraphormerGraphEncoderLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 768,   #hidden_dimension
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,  #注意力头的数目
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        export: bool = False,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
        init_fn: Callable = None,
        pre_layernorm: bool = False,
    ) -> None:
        super().__init__()

        if init_fn is not None:
            init_fn()

        # 初始化参数
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.q_noise = q_noise
        self.qn_block_size = qn_block_size
        self.pre_layernorm = pre_layernorm

        # 初始化dropout模块
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )
        self.activation_dropout_module = FairseqDropout(
            activation_dropout, module_name=self.__class__.__name__
        )

        # 初始化激活函数
        self.activation_fn = utils.get_activation_fn(activation_fn)

        # 构建自注意力层
        self.self_attn = self.build_self_attention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )

        # 自注意力层后的LayerNorm层
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim, export=export)

        # 构建前馈全连接层1
        self.fc1 = self.build_fc1(
            self.embedding_dim,
            ffn_embedding_dim,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )

        # 构建前馈全连接层2
        self.fc2 = self.build_fc2(
            ffn_embedding_dim,
            self.embedding_dim,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )

        # 前馈层后的LayerNorm层
        self.final_layer_norm = LayerNorm(self.embedding_dim, export=export)

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        # 构建前馈全连接层1，并应用量化噪声
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        # 构建前馈全连接层2，并应用量化噪声
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_self_attention(
        self,
        embed_dim,
        num_attention_heads,
        dropout,
        self_attention,
        q_noise,
        qn_block_size,
    ):
        # 构建自注意力层，并应用噪声
        return MultiheadAttention(
            embed_dim,
            num_attention_heads,
            dropout=dropout,
            self_attention=True,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )

    def forward(
        self,
        x: torch.Tensor,
        self_attn_bias: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        # x: T x B x C
        residual = x
        if self.pre_layernorm:
            # 如果配置为在自注意力层之前使用LayerNorm，应用LayerNorm
            x = self.self_attn_layer_norm(x)

        # 自注意力计算
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            attn_bias=self_attn_bias,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)

        # 添加残差连接并进行LayerNorm
        x = residual + x
        if not self.pre_layernorm:
            # 如果配置为在自注意力层之后使用LayerNorm，应用LayerNorm
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.pre_layernorm:
            # 如果配置为在前馈全连接层之前使用LayerNorm，应用LayerNorm
            x = self.final_layer_norm(x)

        # 前馈全连接层1和激活函数计算
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)

        # 前馈全连接层2计算
        x = self.fc2(x)
        x = self.dropout_module(x)

        # 添加残差连接并进行LayerNorm
        x = residual + x
        if not self.pre_layernorm:
            # 如果配置为在前馈全连接层之后使用LayerNorm，应用LayerNorm
            x = self.final_layer_norm(x)

        return x, attn
