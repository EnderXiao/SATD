from typing import List

import torch
import torch.nn as nn
from einops import rearrange
from torch import FloatTensor, LongTensor
from satd.datamodule.gen_symbols_struct_dict import vocab

from satd.model.pos_enc import WordPosEnc
from satd.model.transformer.MAAM import MultiscaleAttentionAggregationModule
from satd.model.transformer.transformer_decoder import (
    TransformerDecoder,
    TransformerDecoderLayer,
)
from satd.utils.generation_utils import DecodeModel

vocab_size = len(vocab)


class Mlp(nn.Module):

    def __init__(self,
                 in_dim,
                 out_dim,
                 hidden_dim,
                 activation=nn.GELU,
                 dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


def _build_transformer_decoder(
        d_model: int,
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        dc: int,
        cross_coverage: bool,
        self_coverage: bool,
) -> nn.TransformerDecoder:
    decoder_layer = TransformerDecoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    )
    if cross_coverage or self_coverage:
        maam = MultiscaleAttentionAggregationModule(nhead, dc, cross_coverage,
                                        self_coverage)
    else:
        maam = None

    decoder = TransformerDecoder(decoder_layer, num_decoder_layers, maam)
    return decoder


class MyDecoder(DecodeModel):

    def __init__(
            self,
            d_model: int,
            nhead: int,
            num_decoder_layers: int,
            dim_feedforward: int,
            dropout: float,
            dc: int,
            cross_coverage: bool,
            self_coverage: bool,
    ):
        super().__init__()
        self.d_model = d_model
        self.struct_vocab = [105, 106, 107, 108, 109, 110, 111]
        self.struct_num = len(self.struct_vocab)
        self.word_embed = nn.Sequential(nn.Embedding(vocab_size, d_model),
                                        nn.LayerNorm(d_model))

        self.pos_enc = WordPosEnc(d_model=d_model)

        self.norm = nn.LayerNorm(d_model)

        self.model_ast = _build_transformer_decoder(
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            dc=dc,
            cross_coverage=False,
            self_coverage=False,
        )

        self.model = _build_transformer_decoder(
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            dc=dc,
            cross_coverage=True,
            self_coverage=True,
        )

        self.proj = nn.Linear(d_model, vocab_size)
        # self.structProj = nn.Linear(d_model, len(self.struct_vocab))

    def _build_attention_mask(self, length):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.full((length, length),
                          fill_value=1,
                          dtype=torch.bool,
                          device=self.device)
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self,
                imageSrc: FloatTensor,
                imageSrc_mask: LongTensor,
                src: FloatTensor,
                src_mask: FloatTensor,
                tgt: LongTensor,
                is_train: bool = True) -> FloatTensor:
        """generate output for tgt

        Parameters
        ----------
        imageSrc : FloatTensor
            [b, h, w, d]
        imageSrc_mask: LongTensor
            [b, h, w]
        src : FloatTensor
            [b, l, voc_size]
        src_mask: FloatTensor
            [b, l]
        tgt : LongTensor
            [b, l]
        struct_tgt : LongTensor
            [n, l, 7]
        is_train : bool

        Returns
        -------
        FloatTensor
            [b, l, vocab_size]
        """
        batch_size, length = tgt.size()
        tgt_mask = self._build_attention_mask(length)
        tgt_pad_mask = tgt == vocab.PAD_IDX
        tgt = self.word_embed(tgt)  # [b, l, d]
        tgt = self.pos_enc(tgt)  # [b, l, d]
        tgt = self.norm(tgt)

        h = src.shape[1]
        tgt = rearrange(tgt, "b l d -> l b d")
        # TODO: 10.8修改输入前形状不匹配的问题
        # src = rearrange(src, "b l d -> l b d")

        astOut = self.model_ast(
            tgt=tgt,
            memory=src,
            height=h,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_mask,
        )
        astOut = rearrange(astOut, "l b d -> b l d")

        img_h = imageSrc.shape[1]
        imageSrc = rearrange(imageSrc, "b h w d -> (h w) b d")
        imageSrc_mask = rearrange(imageSrc_mask, "b h w -> b (h w)")
        astOut = rearrange(astOut, "b l d -> l b d")
        wordOut = self.model(
            tgt=astOut,
            memory=imageSrc,
            height=img_h,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=imageSrc_mask,
        )
        wordOut = rearrange(wordOut, "l b d -> b l d")
        out = self.proj(wordOut)
        return out

    def transform(self,
                  src: List[FloatTensor],
                  src_mask: List[LongTensor],
                  input_ids: LongTensor,
                  tree_tgt: FloatTensor = None,
                  tree_tgt_mask: FloatTensor = None) -> FloatTensor:
        assert len(src) == 1 and len(src_mask) == 1
        word_out = self(src[0],
                        src_mask[0],
                        tree_tgt,
                        tree_tgt_mask,
                        input_ids,
                        is_train=False)
        return word_out
