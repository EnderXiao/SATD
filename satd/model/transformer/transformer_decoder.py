import copy
from functools import partial
from typing import Optional

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .MAAM import MultiscaleAttentionAggregationModule
from .attention import MultiheadAttention
from einops import rearrange


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerDecoder(nn.Module):

    def __init__(
        self,
        decoder_layer,
        num_layers: int,
        maam: Optional[MultiscaleAttentionAggregationModule],
        norm=None,
    ):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

        self.maam = maam

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        height: int,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        need_attn_score: Optional[bool] = False
    ) -> Tensor:
        """Transformer 解码器

        Args:
            tgt (Tensor): 解码器的输入序列 
            memory (Tensor): encoder的输入序列
            height (int): 向量高度
            tgt_mask (Optional[Tensor], optional): 输入序列掩码. Defaults to None.
            memory_mask (Optional[Tensor], optional): encoder输入掩码. Defaults to None.
            tgt_key_padding_mask (Optional[Tensor], optional): 输入向量padding标志位. Defaults to None.
            memory_key_padding_mask (Optional[Tensor], optional): encoder输入向量padding标志位. Defaults to None.
            need_attn_score: 是否需要返回注意力分株

        Returns:
            Tensor: _description_
        """
        # 从初始输入tgt迭代为最终输出
        output = tgt

        # 控制第一个layer没有maam
        maam = None
        for i, mod in enumerate(self.layers):
            output, attn = mod(
                output,
                memory,
                maam,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )  # [l, 2b, d], [2b * num_head, l, src_l]
            # maam通过partial固定参数
            if i != len(self.layers) - 1 and self.maam is not None:
                maam = partial(self.maam, attn, memory_key_padding_mask, height)

        if self.norm is not None:
            output = self.norm(output)
        if need_attn_score:
            return output, attn
        return output


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model,
                                                 nhead,
                                                 dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = F.relu

    def __setstate__(self, state):
        # 读取序列化模型参数
        if "activation" not in state:
            state["activation"] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        maam: Optional[MultiscaleAttentionAggregationModule],
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        # 自注意力机制
        # print("tf decoder self_attn: ", tgt.shape)
        tgt2 = self.self_attn(tgt,
                              tgt,
                              tgt,
                              attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        # [l, 2b, d]
        # add & norm 残差链接层
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # print("tf decoder cross_attn: ", tgt.shape, memory.shape)
        tgt2, attn = self.multihead_attn(
            tgt,
            memory,
            memory,
            maam=maam,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )  # [l, 2b, d], [2b * num_head, l, src_l]
        # add & norm 残差链接层
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        # FFN 逐位前馈层
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        # add & norm 残差链接层
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attn
