import zipfile
from typing import List

import pytorch_lightning as pl
import torch.optim as optim
import torch
import torch.nn as nn
from torch import FloatTensor, LongTensor
# import matplotlib.pyplot as plt

from satd.datamodule.datamodule import vocab
from satd.datamodule.dataset import Batch
from satd.model.SATD import SATD

# from timm.scheduler import create_scheduler
from satd.utils.utils import (ExpRateRecorder, Hypothesis,
                               ce_loss, to_bi_tgt_out)
from satd.utils.generation_utils import my_convert
import torch.nn.functional as F
from einops import rearrange


class LitSATD(pl.LightningModule):

    def __init__(
            self,
            d_model: int,
            # encoder
            growth_rate: int,
            num_layers: int,
            # decoder
            nhead: int,
            num_tree_decoder_layer: int,
            num_decoder_layers: int,
            dim_feedforward: int,
            dropout: float,
            dc: int,
            cross_coverage: bool,
            self_coverage: bool,
            threshold: float,
            # beam search
            beam_size: int,
            max_len: int,
            alpha: float,
            early_stopping: bool,
            temperature: float,
            # training
            learning_rate: float,
            patience: int,
            checkpoint_path: str,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.satd_model = SATD(
            d_model=d_model,
            growth_rate=growth_rate,
            num_layers=num_layers,
            nhead=nhead,
            num_tree_decoder_layer=num_tree_decoder_layer,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            dc=dc,
            cross_coverage=cross_coverage,
            self_coverage=self_coverage,
        )
        self.n_head = nhead
        self.struct_loss = torch.nn.BCELoss()
        self.exprate_recorder = ExpRateRecorder()

    # TODO: 10.8将结构标签加入预测
    def forward(self, img: FloatTensor, img_mask: LongTensor,
                childrenTgt: LongTensor, parentTgt: LongTensor, is_train: bool = True) -> FloatTensor:
        """run img and bi-tgt

        Parameters
        ----------
        img : FloatTensor
            [b, 1, h, w]
        img_mask: LongTensor
            [b, h, w]
        childrenTgt : LongTensor
            [2b, l]
        parentTgt : LongTensor
            [2b, l]
        Returns
        -------
        FloatTensor
            [2b, l, vocab_size]
        """
        return self.satd_model(img, img_mask, childrenTgt, parentTgt, is_train)

    def view_model(self):
        print("Model Layers:")
        for layer in self.satd_model.state_dict():
            print(layer, self.satd_model.state_dict()[layer].shape)

    def view_checkpoint(self, checkpoint_path):
        print("Checkpoint Layers:")
        pretrained = checkpoint_path
        pretrained_dict = torch.load(pretrained, map_location='cpu')
        for k in pretrained_dict["state_dict"]:
            print(k, pretrained_dict["state_dict"][k].shape)

    def on_fit_start(self):
        print("Fit Started")
        # # 加载模型的权重
        # checkpoint = torch.load('lightning_logs/version_0/checkpoints/epoch=203-step=225215-val_ExpRate=0.6294.ckpt')
        #
        # # 获取模型的 state_dict
        # state_dict = checkpoint['state_dict']
        #
        # # 遍历 state_dict 并替换层级名称
        # new_state_dict = {}
        # for key, value in state_dict.items():
        #     if key.startswith('comer_model'):
        #         new_key = key.replace('comer_model', 'satd_model')
        #         new_state_dict[new_key] = value
        #     else:
        #         new_state_dict[key] = value



    def on_train_start(self):
        print("Train Started")

    def cal_kl_loss(self, child_alphas, parent_alphas, labels, num_head):
        # TODO: 重排注意力分数将通道分开
        child_alphas = rearrange(child_alphas, "(b h) l t -> b l h t", h=num_head)
        parent_alphas = rearrange(parent_alphas, "(b h) l t -> b l h t", h=num_head)
        batch_size, len1, num_head, len2 = child_alphas.shape
        batch_size_half = batch_size // 2

        # TODO: 初始化父子关系序列
        parent_ids = torch.full(
            (batch_size // 2, len1),
            fill_value=vocab.PAD_IDX,
            dtype=torch.long,
            device=child_alphas.device,
        )
        parent_ids_reverse = torch.full(
            (batch_size // 2, len1),
            fill_value=vocab.PAD_IDX,
            dtype=torch.long,
            device=child_alphas.device,
        )

        # TODO: 从标签中获取id序列，并且构造一个反向序列
        parent_ids[:, 1:] = labels[:, :, 2].to(child_alphas.device)

        reverse_tmp = [torch.flip(t, dims=[0]) for t in parent_ids]

        # TODO: 将pad掉的id序列设为其标号以保证父子相同
        for b in range(0, batch_size_half):
            for i in range(0, len1):
                t = parent_ids[b][i]
                if t == torch.tensor(0, dtype=t.dtype):
                    parent_ids[b][i] = torch.tensor(i, dtype=t.dtype)

        # TODO: 反向id序列需要将pad标签特殊处理掉
        for b in range(0, batch_size_half):
            id_index = 1
            for i in range(0, len1):
                t = reverse_tmp[b][i]
                if t != torch.tensor(0, dtype=t.dtype):
                    parent_ids_reverse[b][id_index] = t
                    id_index += 1
        for b in range(0, batch_size_half):
            for i in range(0, len1):
                t = parent_ids_reverse[b][i]
                if t == torch.tensor(0, dtype=t.dtype) and i != 0:
                    parent_ids_reverse[b][i] = torch.tensor(i, dtype=t.dtype)

        # TODO: 获取注意力分数，需要将第一个<sos>的注意力去除
        new_child_alphas = torch.zeros((batch_size_half, len1, num_head, len2)).to(child_alphas.device)
        new_child_reverse = torch.zeros((batch_size_half, len1, num_head, len2)).to(child_alphas.device)
        new_child_alphas[:, 1:, :, :] = child_alphas[:batch_size_half, 1:, :, :].clone()
        new_child_reverse[:, 1:, :, :] = child_alphas[batch_size_half:, 1:, :, :].clone()

        # TODO: 将batch和len压缩，便于之后按id取数
        new_child_alphas = new_child_alphas.view((batch_size_half * len1, num_head, len2))
        new_child_reverse = new_child_reverse.view((batch_size_half * len1, num_head, len2))

        # TODO: 计算压缩之后的id
        parent_ids = parent_ids + len1 * torch.arange(batch_size_half)[:, None].to(parent_ids.device)
        parent_ids_reverse = parent_ids_reverse + len1 * torch.arange(batch_size_half)[:, None].to(parent_ids.device)

        # TODO: 根据父子关系取出对应的子注意力分数
        new_child_alphas = new_child_alphas[parent_ids]
        new_child_reverse = new_child_reverse[parent_ids_reverse]

        new_child_alphas = new_child_alphas.view((batch_size_half, len1, num_head, len2))[:, 1:, :, :]
        new_child_reverse = new_child_reverse.view((batch_size_half, len1, num_head, len2))[:, 1:, :, :]

        # TODO: 变回原样
        new_child_alphas = rearrange(new_child_alphas, "b l h t -> (b h) l t")
        new_child_reverse = rearrange(new_child_reverse, "b l h t -> (b h) l t")

        # TODO: 取出父注意力分数
        new_parent_alphas = parent_alphas[:batch_size_half, 1:, :, :].clone()
        new_parent_alphas_reverse = parent_alphas[batch_size_half:, 1:, :, :].clone()

        new_parent_alphas = rearrange(new_parent_alphas, "b l h t -> (b h) l t")
        new_parent_alphas_reverse = rearrange(new_parent_alphas_reverse, "b l h t -> (b h) l t")

        # TODO: 计算正向KL散度
        T = 4
        KL_loss = torch.nn.KLDivLoss(reduction='batchmean').cuda()
        pred_kl_loss = KL_loss(F.log_softmax(new_child_alphas / T, dim=-1),
                               F.softmax(new_parent_alphas / T, dim=-1)) * T * T

        # TODO: 计算反向KL散度
        KL_loss_reverse = torch.nn.KLDivLoss(reduction='batchmean').cuda()
        pred_kl_reverse_loss = KL_loss_reverse(F.log_softmax(new_child_reverse / T, dim=-1),
                                               F.softmax(new_parent_alphas_reverse / T, dim=-1)) * T * T

        # KL_alpha = new_child_alphas * (torch.log(new_child_alphas + 1e-10) - torch.log(new_parent_alphas + 1e-10)) * image_mask
        # KL_loss = (KL_alpha.sum(-1).sum(-1) * label_mask[:, :-1, 0]).sum(-1).sum(-1) / (label_mask.sum() - batch_size)

        return pred_kl_loss + pred_kl_reverse_loss

    def training_step(self, batch: Batch, _):
        # tgt为带开始符和PAD的label, out为带结束符和PAD的label
        # print()
        # print("Training Step")

        childrenLabel = batch.indices[:, :, 1]
        parentLabel = batch.indices[:, :, 3]

        childrenTgt, childrenOut = to_bi_tgt_out(childrenLabel, self.device)
        parentTgt, parentOut = to_bi_tgt_out(parentLabel, self.device)
        out_parent_hat, tree_attn, word_attn = self(batch.imgs, batch.mask, childrenTgt,
                                                    parentTgt, is_train=True)

        # labelList = [vocab.indices2label(x) for x in parentOut.tolist()]
        # for label in labelList:
        #     print(label)
        parent_loss = ce_loss(out_parent_hat, parentOut)
        loss = parent_loss

        # TODO: KL散度
        # T = 4
        beta = 1.0
        # KL_loss = torch.nn.KLDivLoss(reduction='batchmean').cuda()
        # pred_kl_loss = KL_loss(F.log_softmax(tree_attn / T, dim=-1), F.softmax(word_attn / T, dim=-1)) * T * T
        pred_kl_loss = self.cal_kl_loss(tree_attn, word_attn, labels=batch.indices, num_head=self.n_head)
        loss += pred_kl_loss * beta

        # print("Struct Loss: ", struct_average_loss)
        # print("child_loss Loss: ", child_loss)
        # x = input()

        self.log("train_loss",
                 loss,
                 on_step=False,
                 on_epoch=True,
                 sync_dist=True)

        self.log("kl_loss",
                 pred_kl_loss,
                 on_step=False,
                 on_epoch=True,
                 sync_dist=True)

        self.log("word_loss",
                 parent_loss,
                 on_step=False,
                 on_epoch=True,
                 sync_dist=True)

        return loss

    def validation_step(self, batch: Batch, _):
        # print(len(batch))

        labelIndices = batch.indices[:, :, 0]
        childrenLabel = batch.indices[:, :, 1]
        parentLabel = batch.indices[:, :, 3]

        childrenTgt, childrenOut = to_bi_tgt_out(childrenLabel, self.device)
        parentTgt, parentOut = to_bi_tgt_out(parentLabel, self.device)

        out_parent_hat, tree_attn, word_attn = self(batch.imgs, batch.mask, childrenTgt,
                                                    parentTgt, is_train=True)

        parent_loss = ce_loss(out_parent_hat, parentOut)

        loss = parent_loss

        # TODO: KL散度
        # T = 4
        beta = 1.0
        # KL_loss = torch.nn.KLDivLoss(reduction='batchmean').cuda()
        # pred_kl_loss = KL_loss(F.log_softmax(tree_attn / T, dim=-1), F.softmax(word_attn / T, dim=-1)) * T * T
        pred_kl_loss = self.cal_kl_loss(tree_attn, word_attn, labels=batch.indices, num_head=self.n_head)
        loss += pred_kl_loss * beta

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        # TODO: handle Exp Rate
        hyps, tree_attn, word_attn = self.approximate_joint_search(batch.imgs, batch.mask)
        # print("validation end hyps: ", len(hyps))
        self.exprate_recorder([h.seq for h in hyps], batch.indices)
        self.log(
            "val_ExpRate",
            self.exprate_recorder,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

    def test_step(self, batch: Batch, _):

        hyps, tree_attn, word_attn = self.approximate_joint_search(
            batch.imgs, batch.mask)

        self.exprate_recorder([h.seq for h in hyps], batch.indices)

        # return batch.img_bases, [
        #     vocab.indices2label(h.seq) for h in hyps
        # ]
        return batch.img_bases, [h.seq for h in hyps], tree_attn, word_attn

    def test_epoch_end(self, test_outputs) -> None:
        exprate = self.exprate_recorder.compute()
        print(f"Validation ExpRate: {exprate}")

        print(f"length of total file: {len(test_outputs)}")
        result_convert_str = []
        with open("result.txt", "wb") as fa:
            for img_bases, preds, _, _ in test_outputs:
                tmp_list = []
                for img_base, pred in zip(img_bases, preds):
                    result_str = my_convert(pred)
                    pred = vocab.indices2label(pred)
                    tmp_list.append((img_base, result_str))
                    content = f"{img_base} {pred} \n".encode()
                    fa.write(content)
                result_convert_str.append(tmp_list)
        fa.close()

        with open("result_convert.txt", "wb") as fa:
            for result in result_convert_str:
                for img_base, pred in result:
                    content = f"{img_base} {pred} \n".encode()
                    fa.write(content)
        fa.close()

        with zipfile.ZipFile("result.zip", "w") as zip_f:
            for img_bases, preds, _, _ in test_outputs:
                for img_base, pred in zip(img_bases, preds):
                    content = f"%{img_base}\n${pred}$".encode()
                    with zip_f.open(f"{img_base}.txt", "w") as f:
                        f.write(content)

    def approximate_joint_search(self, img: FloatTensor,
                                 mask: LongTensor) -> List[Hypothesis]:
        return self.satd_model.beam_search(img, mask, **self.hparams)

    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.parameters(),
            lr=self.hparams.learning_rate,
            momentum=0.9,
            weight_decay=1e-4,
        )

        reduce_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.25,
            patience=self.hparams.patience //
                     self.trainer.check_val_every_n_epoch,
        )
        # cosin_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     optimizer, T_0=5, T_mult=1, eta_min=0.001, last_epoch=-1)
        # timm_scheduler, num_epochs = create_scheduler(arg, optimizer)

        scheduler = {
            "scheduler": reduce_scheduler,
            "monitor": "val_ExpRate",
            "interval": "epoch",
            "frequency": self.trainer.check_val_every_n_epoch,
            "strict": True,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
