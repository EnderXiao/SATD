from typing import List
import pytorch_lightning as pl
import torch
from torch import FloatTensor, LongTensor
import scipy
import numpy as np
import os
import logging

# import sys

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from satd.utils.utils import Hypothesis

from .myDecoder import MyDecoder
from .TreeDecoder import TreeDecoder

from .encoder import Encoder
from satd.datamodule.gen_symbols_struct_dict import vocab


vocab_size = len(vocab)


class SATD(pl.LightningModule):

    def __init__(
            self,
            d_model: int,
            growth_rate: int,
            num_layers: int,
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

        self.encoder = Encoder(d_model=d_model,
                               growth_rate=growth_rate,
                               num_layers=num_layers,
                               use_vision_attention=False)

        self.decoder = TreeDecoder(
            d_model=d_model,
            nhead=nhead,
            num_tree_decoder_layer=num_tree_decoder_layer,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            dc=dc,
            cross_coverage=cross_coverage,
            self_coverage=self_coverage,
        )

    def forward(self,
                img: FloatTensor,
                img_mask: LongTensor,
                children_tgt: LongTensor,
                parent_tgt: LongTensor,
                is_train: bool = True) -> [FloatTensor, FloatTensor]:
        """run img and bi-tgt

        Parameters
        ----------
        img : FloatTensor
            [b, 1, h, w]
        img_mask: LongTensor
            [b, h, w]
        children_tgt : LongTensor
            [2b, l]
        parent_tgt : LongTensor
            [2b, l]
        is_train: bool = True
        Returns
        -------
        FloatTensor
            [2b, l, vocab_size]
        """
        feature, mask = self.encoder(img, img_mask)  # [b, h, w, c] [b, h, w]
        feature = torch.cat((feature, feature), dim=0)  # [2b, h, w, c]
        mask = torch.cat((mask, mask), dim=0)  # [2b, h, w]

        parent_out, tree_attn, word_attn = self.decoder(feature,
                                                        mask,
                                                        children_tgt,
                                                        parent_tgt,
                                                        need_attn=True)
        # parent_out [2b, l, vocab_size]
        # tree_attn [2b * num_head, l, t]
        # word_attn [2b * num_head, l, t]
        # print("After childrenOut:", childrenOut.shape)
        # x = input()
        if is_train:
            return parent_out, tree_attn, word_attn
        else:
            return parent_out
        # out = self.decoder(feature, mask, tgt)

        # return out

    def beam_search(
            self,
            img: FloatTensor,
            img_mask: LongTensor,
            beam_size: int,
            max_len: int,
            alpha: float,
            early_stopping: bool,
            temperature: float,
            **kwargs,
    ) -> List[Hypothesis]:
        """run bi-direction beam search for given img

        Parameters
        ----------
        img : FloatTensor
            [b, 1, h', w']
        img_mask: LongTensor
            [b, h', w']
        beam_size : int
        max_len : int
        alpha: float,
        early_stopping: bool,
        temperature: float,

        Returns
        -------
        List[Hypothesis]
        """
        # freeze the encoder parameters
        # self.encoder.eval()
        # with torch.no_grad():
        feature, mask = self.encoder(img, img_mask)  # [b, t, d]
        hyps, tree_attn, word_attn  = self.decoder.beam_search([feature], [mask], beam_size,
                                        max_len, alpha, early_stopping,
                                        temperature)
        return hyps, tree_attn, word_attn
