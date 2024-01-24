from typing import List, Optional, Tuple, Union

import torch
import yaml
import math
from torch import nn
import torch.nn.functional as F
from satd.datamodule.gen_symbols_struct_dict import vocab
from einops import rearrange
from torch import LongTensor
from torchmetrics import Metric
from difflib import SequenceMatcher

vocab_size = len(vocab)


class GroupWiseLinear(nn.Module):
    # could be changed to:
    # output = torch.einsum('ijk,zjk->ij', x, self.W)
    # or output = torch.einsum('ijk,jk->ij', x, self.W[0])
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x


class Hypothesis:
    seq: List[int]
    score: float

    def __init__(
            self,
            seq_tensor: LongTensor,
            score: float,
            direction: str,
    ) -> None:
        assert direction in {"l2r", "r2l"}
        raw_seq = seq_tensor.tolist()

        if direction == "r2l":
            result = raw_seq[::-1]
        else:
            result = raw_seq

        self.seq = result
        self.score = score

    def __len__(self):
        if len(self.seq) != 0:
            return len(self.seq)
        else:
            return 1

    def __str__(self):
        return f"seq: {self.seq}, score: {self.score}"


class ExpRateRecorder(Metric):

    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("total_line",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        self.add_state("rec", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, indices_hat: List[List[int]], indices: List[List[int]]):
        word_indices = indices[:, :, 3]
        for word_pred, word_truth in zip(indices_hat, word_indices):
            # result_str = my_convert(word_pred)
            word_pred = vocab.indices2label(word_pred)
            word_truth = vocab.indices2label(word_truth)
            print("word pred: ", word_pred)
            # print("word res: ", result_str)
            print("word truth: ", word_truth)

            is_same = word_pred == word_truth
            if is_same:
                print("batch correct:")
                print("word pred: ", word_pred)
                print("word truth: ", word_truth)
                self.rec += 1
            else:
                check_twice = 1
                if len(word_pred) == len(word_truth) + 1:
                    for pred_i, truth_i in zip(word_pred, word_truth):
                        if pred_i != truth_i:
                            check_twice = 0
                    if check_twice and word_pred[-1] == 'right':
                        print("batch correct with right:")
                        print("word pred: ", word_pred)
                        print("word truth: ", word_truth)
                        self.rec += 1
            self.total_line += 1

    def compute(self) -> float:
        exp_rate = self.rec / self.total_line
        return exp_rate


def ce_loss(
        output_hat: torch.Tensor,
        output: torch.Tensor,
        ignore_idx: int = vocab.PAD_IDX,
        reduction: str = "mean",
) -> torch.Tensor:
    """comput cross-entropy loss

    Args:
        output_hat (torch.Tensor): [batch, len, e]
        output (torch.Tensor): [batch, len]
        ignore_idx (int):

    Returns:
        torch.Tensor: loss value
    """
    flat_hat = rearrange(output_hat, "b l e -> (b l) e")
    flat = rearrange(output, "b l -> (b l)")
    loss = F.cross_entropy(flat_hat,
                           flat,
                           ignore_index=ignore_idx,
                           reduction=reduction)
    return loss


def to_tgt_output(
        tokens: List[List[int]],
        direction: str,
        device: torch.device,
        pad_to_len: Optional[int] = None) -> Tuple[LongTensor, LongTensor]:
    """Generate tgt and out for indices

    Parameters
    ----------
    tokens : Union[List[List[int]], List[LongTensor]]
        indices: [b, l]
    direction : str
        one of "l2f" and "r2l"
    device : torch.device

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        tgt, out: [b, l], [b, l]
    """
    assert direction in {"l2r", "r2l"}

    if isinstance(tokens[0], list):
        tokens = [torch.tensor(t, dtype=torch.long) for t in tokens]

    if direction == "l2r":
        tokens = tokens
        start_w = vocab.SOS_IDX
        stop_w = vocab.EOS_IDX
    else:
        batch_size = len(tokens)
        max_len = len(tokens[0])
        tokens_zero = []
        tokens = [torch.flip(t, dims=[0]) for t in tokens]
        for b in range(0, batch_size):
            tmp_token = torch.zeros(max_len)
            token_index = 0
            for i in range(0, max_len):
                t = tokens[b][i]
                if t != torch.tensor(0, dtype=t.dtype):
                    tmp_token[token_index] = t
                    token_index += 1
                else:
                    continue
            tokens_zero.append(tmp_token)
        tokens = tokens_zero
        start_w = vocab.EOS_IDX
        stop_w = vocab.SOS_IDX

    batch_size = len(tokens)
    lens = [len(t) for t in tokens]

    length = max(lens) + 1
    if pad_to_len is not None:
        length = max(length, pad_to_len)

    tgt = torch.full(
        (batch_size, length),
        fill_value=vocab.PAD_IDX,
        dtype=torch.long,
        device=device,
    )
    out = torch.full(
        (batch_size, length),
        fill_value=vocab.PAD_IDX,
        dtype=torch.long,
        device=device,
    )

    for i, token in enumerate(tokens):
        tgt[i, 0] = start_w
        tgt[i, 1:(1 + lens[i])] = token

        out[i, :lens[i]] = token
        out[i, lens[i]] = stop_w

    return tgt, out


def to_bi_tgt_out(
        tokens: List[List[int]],
        device: torch.device) -> Tuple[LongTensor, LongTensor]:
    """Generate bidirection tgt and out

    Parameters
    ----------
    tokens : List[List[int]]
        indices: [b, l]
    device : torch.device

    Returns
    -------
    Tuple[LongTensor, LongTensor]
        tgt, out: [2b, l], [2b, l]
    """
    l2r_tgt, l2r_out = to_tgt_output(tokens,
                                     "l2r",
                                     device)
    r2l_tgt, r2l_out = to_tgt_output(tokens,
                                     "r2l",
                                     device)

    tgt = torch.cat((l2r_tgt, r2l_tgt), dim=0)
    out = torch.cat((l2r_out, r2l_out), dim=0)

    return tgt, out


def my_convert(gtd_list):
    struct_stack = []
    result_str = []
    struct_list = ["above", "below", "sub", "sup", "l-sup", "inside", "right"]
    for i in range(len(gtd_list)):
        word_now = vocab.idx2word[gtd_list[i]]
        # print("convert: ", word_now)
        if word_now not in struct_list:
            result_str += [word_now]
        else:
            tmp_word = '}'
            if word_now == "right":
                if len(struct_stack) != 0:
                    struct = struct_stack.pop()
                    if struct == "l-sup":
                        result_str += [']']
                    else:
                        result_str += ['}']
                else:
                    continue
            elif word_now == "below":
                if len(struct_stack) != 0:
                    while len(struct_stack):
                        struct = struct_stack.pop()
                        if struct == "above":
                            result_str += ['}']
                        elif struct == "l-sup":
                            result_str += [']']
                        else:
                            result_str += ['}']
                    result_str += ['{']
                else:
                    result_str += ['{']
                struct_stack.append(word_now)
            else:
                if word_now == "sup":
                    if len(struct_stack) != 0:
                        if struct_stack[-1] == "sub":
                            struct_stack.pop()
                            result_str += ['}']
                    if result_str[-1] in ["\\sum", "\\lim", "\\int"]:
                        result_str += ['limits', '^', '{']
                    else:
                        result_str += ['^', '{']
                elif word_now == "sub":
                    if len(struct_stack) != 0:
                        if struct_stack[-1] == "sup":
                            struct_stack.pop()
                            result_str += ['}']
                    if result_str[-1] in ["\\sum", "\\lim", "\\int"]:
                        result_str += ['\\limits', '_', '{']
                    else:
                        result_str += ['_', '{']
                elif word_now == "l-sup":
                    result_str += ['[']
                else:
                    result_str += ['{']
                struct_stack.append(word_now)
    while len(struct_stack) != 0:
        struct = struct_stack.pop()
        if struct == "l-sup":
            result_str += [']']
        else:
            result_str += ['}']
    return " ".join(result_str)


def load_config(yaml_path):
    try:
        with open(yaml_path, 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
    except Exception:
        print('try UTF-8 encoding')
        with open(yaml_path, 'r', encoding='UTF-8') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)

    params = params['data']

    # if not params['experiment']:
    #     print('expriment name cannot be empty!')
    #     exit(-1)

    if not params['train_image_path']:
        print('training images cannot be empty!')
        exit(-1)

    if not params['train_label_path']:
        print('training labels cannot be empty!')
        exit(-1)

    if not params['eval_image_path']:
        print('test images cannot be empty!')
        exit(-1)

    if not params['eval_label_path']:
        print('test labels cannot be empty!')
        exit(-1)

    if not params['word_path']:
        print('word dict cannot be empty')
        exit(-1)
    return params
