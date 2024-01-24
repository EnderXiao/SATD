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


class TreeDecoder(DecodeModel):

    def __init__(
            self,
            d_model: int,
            nhead: int,
            num_tree_decoder_layer: int,
            num_decoder_layers: int,
            dim_feedforward: int,
            dropout: float,
            dc: int,
            cross_coverage: bool,
            self_coverage: bool,
    ):
        super().__init__()

        self.word_embed = nn.Sequential(nn.Embedding(vocab_size, d_model),
                                        nn.LayerNorm(d_model))

        self.pos_enc = WordPosEnc(d_model=d_model)

        self.norm = nn.LayerNorm(d_model)

        self.tree_decoder = _build_transformer_decoder(
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_tree_decoder_layer,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            dc=dc,
            cross_coverage=cross_coverage,
            self_coverage=self_coverage,
        )

        self.proj = nn.Linear(d_model, d_model)

        # TODO: 讲两个Decoder整合为一个
        self.d_model = d_model
        self.struct_vocab = [105, 106, 107, 108, 109, 110, 111]
        self.struct_num = len(self.struct_vocab)

        self.norm2 = nn.LayerNorm(d_model)

        self.ast_decoder = _build_transformer_decoder(
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=3,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            dc=dc,
            cross_coverage=False,
            self_coverage=False,
        )

        self.word_decoder = _build_transformer_decoder(
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            dc=dc,
            cross_coverage=True,
            self_coverage=True,
        )

        self.word_proj = nn.Linear(d_model, vocab_size)

    def _build_attention_mask(self, length):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.full((length, length),
                          fill_value=1,
                          dtype=torch.bool,
                          device=self.device)
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, src: FloatTensor, src_mask: LongTensor,
                tgt: LongTensor, tgt2: LongTensor, need_attn: bool=False) -> [FloatTensor, FloatTensor]:
        """generate output for tgt

        Parameters
        ----------
        src : FloatTensor
            [2b, h, w, d]
        src_mask: LongTensor
            [2b, h, w]
        tgt : LongTensor
            [2b, l]
        tgt2 : LongTensor
            [2b, l]
        Returns
        -------
        FloatTensor
            [b, l, vocab_size]
        """
        _, length = tgt.size()
        tgt_mask = self._build_attention_mask(length)
        tgt_pad_mask = tgt == vocab.PAD_IDX

        # print("TreeDecoder tgt after proceed: ", tgt.shape)
        tgt = self.word_embed(tgt)  # [2b, l, norm(d)] normalized final dimension
        tgt = self.pos_enc(tgt)  # [2b, l, plus_pos_emb(d)] plus position embedding
        tgt = self.norm(tgt)  # [2b, l, norm(d)]

        # print("TreeDecoder Input ImgShape: ", src.shape)
        # print("TreeDecoder Input MaskShape: ", src_mask.shape)
        # print("TreeDecoder Input TgtShape: ", tgt.shape)
        h = src.shape[1]
        src = rearrange(src, "b h w d -> (h w) b d")
        src_mask = rearrange(src_mask, "b h w -> b (h w)")
        tgt = rearrange(tgt, "b l d -> l b d")

        tree_out, tree_attn = self.tree_decoder(
            tgt=tgt,
            memory=src,
            height=h,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_mask,
            need_attn_score=True
        )

        tree_out = rearrange(tree_out, "l b d -> b l d")
        tree_out = self.proj(tree_out)

        batch_size, length = tgt2.size()
        tgt2_mask = self._build_attention_mask(length)
        tgt2_pad_mask = tgt2 == vocab.PAD_IDX
        tgt2 = self.word_embed(tgt2)  # [b, l, d]
        tgt2 = self.pos_enc(tgt2)  # [b, l, d]
        tgt2 = self.norm2(tgt2)

        tree_h = tree_out.shape[1]
        tgt2 = rearrange(tgt2, "b l d -> l b d")
        # TODO: 10.8修改输入前形状不匹配的问题
        # tree_out = rearrange(src, "b l d -> l b d")

        astOut = self.ast_decoder(
            tgt=tgt2,
            memory=tree_out,
            height=tree_h,
            tgt_mask=tgt2_mask,
            tgt_key_padding_mask=tgt2_pad_mask,
            memory_key_padding_mask=tgt_pad_mask,
        )
        astOut = rearrange(astOut, "l b d -> b l d")

        astOut = rearrange(astOut, "b l d -> l b d")
        wordOut, word_attn = self.word_decoder(
            tgt=astOut,
            memory=src,
            height=h,
            tgt_mask=tgt2_mask,
            tgt_key_padding_mask=tgt2_pad_mask,
            memory_key_padding_mask=src_mask,
            need_attn_score=True
        )
        wordOut = rearrange(wordOut, "l b d -> b l d")
        word_out = self.word_proj(wordOut)

        if need_attn:
            return word_out, tree_attn, word_attn
        else:
            return word_out

    def transform(self,
                  src: List[FloatTensor],
                  src_mask: List[LongTensor],
                  input_ids: LongTensor,
                  input_ids2: LongTensor) -> FloatTensor:
        assert len(src) == 1 and len(src_mask) == 1
        word_out, tree_attn, word_attn = self(src[0], src_mask[0], input_ids, input_ids2, need_attn=True)
        return word_out, tree_attn, word_attn
