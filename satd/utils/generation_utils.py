from abc import abstractmethod
from typing import List, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from satd.datamodule.gen_symbols_struct_dict import vocab
from satd.utils.utils import Hypothesis, ce_loss, to_tgt_output
from einops import rearrange
from einops.einops import repeat
from torch import FloatTensor, LongTensor

from .beam_search import BeamSearchScorer
import pdb

# modified from
# https://github.com/huggingface/transformers/blob/af6e01c5bc39467f1e3ce47a2135fb1777af1db2/src/transformers/generation_utils.py#L1843

vocab_size = len(vocab)


class DecodeModel(pl.LightningModule):

    @abstractmethod
    def transform(self, src: List[FloatTensor], src_mask: List[LongTensor],
                  input_ids: LongTensor, input_ids2: LongTensor) -> FloatTensor:
        """decode one step

        Parameters
        ----------
        src : List[FloatTensor]
            [b, t, d]
        src_mask : List[LongTensor]
            [b, t]
        input_ids : LongTensor
            [b, l]
        input_ids2: LongTensor
            [b, l]
        Returns
        -------
        FloatTensor
            [b, l, vocab_size]
        """
        raise NotImplementedError("This is an abstract method.")

    def beam_search(
            self,
            src: List[FloatTensor],
            src_mask: List[LongTensor],
            beam_size: int,
            max_len: int,
            alpha: float,
            early_stopping: bool,
            temperature: float,
    ) -> List[Hypothesis]:
        """run beam search to decode

        Parameters
        ----------
        src : List[FloatTensor]
            [b, t, d]
        src_mask : List[LongTensor]
            [b, t]
        parentTgt : List[LongTensor]
            [b, t, d]
        beam_size : int
        max_len : int
        alpha : float
        early_stopping : bool

        Returns
        -------
        List[Hypothesis]: [batch_size,]
        """
        batch_size = src[0].shape[0] * 2  # mul 2 for bi-direction
        batch_beam_size = batch_size * beam_size
        half_bb_size = batch_beam_size // 2

        for i in range(len(src)):
            # [2 * b, t, d], [l2r l2r, r2l r2l]
            src[i] = torch.cat((src[i], src[i]), dim=0)
            src_mask[i] = torch.cat((src_mask[i], src_mask[i]), dim=0)

        # print("BeamSearch src: ", len(src), src[0].shape)
        # print("BeamSearch src_mask: ", len(src_mask), src_mask[0].shape)
        # print("BeamSearch parentTgt: ", len(parentTgt), parentTgt[0].shape)

        l2r = torch.full(
            (batch_size // 2, 1),
            fill_value=vocab.SOS_IDX,
            dtype=torch.long,
            device=self.device,
        )
        r2l = torch.full(
            (batch_size // 2, 1),
            fill_value=vocab.EOS_IDX,
            dtype=torch.long,
            device=self.device,
        )
        input_ids = torch.cat((l2r, r2l), dim=0)
        beam_scorer = BeamSearchScorer(batch_size, beam_size, alpha,
                                       early_stopping, self.device)

        # first beam search
        hyps, scores = self._beam_search(
            src=src,
            src_mask=src_mask,
            input_ids=input_ids,
            beam_scorer=beam_scorer,
            beam_size=beam_size,
            max_len=max_len,
            temperature=temperature,
        )

        # reverse half last
        # TODO: 将后半猜测反向
        for i in range(half_bb_size, batch_beam_size):
            hyps[i] = torch.flip(hyps[i], dims=[0])

        lens = [len(h) + 1 for h in hyps]  # plus to append start token
        r2l_tgt, r2l_out = to_tgt_output(hyps[:half_bb_size],
                                         "r2l",
                                         self.device,
                                         pad_to_len=max(lens))
        l2r_tgt, l2r_out = to_tgt_output(hyps[half_bb_size:],
                                         "l2r",
                                         self.device,
                                         pad_to_len=max(lens))
        tgt = torch.cat((l2r_tgt, r2l_tgt), dim=0)
        out = torch.cat((l2r_out, r2l_out), dim=0)

        # calculate final score
        rev_scores, tree_attn, word_attn = self._rate(src,
                                src_mask,
                                tgt,
                                out,
                                alpha,
                                temperature)
        rev_scores = torch.cat(
            (rev_scores[half_bb_size:], rev_scores[:half_bb_size]), dim=0)
        scores = scores + rev_scores
        # [2 * b, beam_size]
        scores = rearrange(scores, "(b m) -> b m", b=batch_size)
        # TODO: 使用chunk将l2r和r2l分开
        l2r_scores, r2l_scores = torch.chunk(scores, 2, dim=0)
        # [b, 2 * beam_size]
        scores = torch.cat((l2r_scores, r2l_scores), dim=1)
        # [batch_size, ]
        best_scores, best_indices = torch.max(scores, dim=1)
        best_split = best_indices // beam_size
        best_indices = best_indices % beam_size
        batch_indices = torch.arange(0,
                                     batch_size // 2,
                                     dtype=torch.long,
                                     device=self.device)
        best_indices = (best_split * half_bb_size + batch_indices * beam_size +
                        best_indices)
        # print("beam_search", struct_hyps)
        ret: List[Hypothesis] = []
        tree_attn_list: List[LongTensor] = []
        word_attn_list: List[LongTensor] = []
        tree_attn = rearrange(tree_attn, "(b n) t l -> b n t l", n=8)
        word_attn = rearrange(word_attn, "(b n) t l -> b n t l", n=8)
        for i in range(half_bb_size, batch_beam_size):
            hyp_len = hyps[i].shape[0]
            # print("gen util No i hpy: ", i,  hyps[i].shape)
            # tree_attn[i, :, :hyp_len, :] = torch.flip(tree_attn[i, :, :hyp_len, :], dims=[1])
            word_attn[i, :, :hyp_len, :] = torch.flip(word_attn[i, :, :hyp_len, :], dims=[1])
            # print("gen util attn: ", tree_attn[i].shape)
            # print("gen util attn: ", word_attn[i].shape)
        tree_attn = rearrange(tree_attn, "b n t l -> (b n) t l", n=8)
        word_attn = rearrange(word_attn, "b n t l -> (b n) t l", n=8)
        tree_attn = rearrange(tree_attn, "(b n) t (h w) -> b n t h w", n=8, h=src[0].shape[1])
        word_attn = rearrange(word_attn, "(b n) t (h w) -> b n t h w", n=8, h=src[0].shape[1])
        for idx, score in zip(best_indices, best_scores):
            hpy = Hypothesis(hyps[idx], score, "l2r")
            ret.append(hpy)
            tree_attn_list.append(tree_attn[idx, :, :, :, :])
            word_attn_list.append(word_attn[idx, :, :, :, :])
        return ret, tree_attn_list, word_attn_list

    def _beam_search(
            self,
            src: List[FloatTensor],
            src_mask: List[LongTensor],
            input_ids: LongTensor,
            beam_scorer: BeamSearchScorer,
            beam_size: int,
            max_len: int,
            temperature: float,
            threshold: float = 0.5
    ) -> Tuple[List[LongTensor], FloatTensor]:
        """inner beam search

        Parameters
        ----------
        src : List[FloatTensor]
            [b, t, d]
        src_mask : List[LongTensor]
            [b, t]
        input_ids: LongTensor
            [b, 1]
        beam_size : int
        max_len : int

        Returns
        _______
        Tuple[List[LongTensor], FloatTensor]
            List[LongTensor]: [b * beam_size] without SOS or EOS token
            FloatTensor: [b * beam_size] corresponding scores
        """
        batch_size, cur_len = input_ids.shape

        beam_scores = torch.zeros(batch_size,
                                  dtype=torch.float,
                                  device=self.device)

        while cur_len < max_len and not beam_scorer.is_done():

            # print("gen_util BeamSearch src: ", len(src), src[0].shape)
            # print("gen_util BeamSearch src_mask: ", len(src_mask), src_mask[0].shape)
            # print("gen_util BeamSearch parentTgt: ", len(parentTgt), parentTgt[0].shape)
            word_out, _, _ = self.transform(src, src_mask, input_ids, input_ids)
            # print("gen_util BeamSearch word_out: shape value", word_out.shape, word_out)
            # print("gen_util BeamSearch struct_out: ", struct_out.shape)
            next_token_logits = (word_out[:, -1, :] / temperature)
            # print("next_token_logits: ", next_token_logits.shape)
            # [b *, l, v]
            next_token_scores = F.log_softmax(next_token_logits, dim=-1)
            # print("Loop add beam_scores shape & value: ", beam_scores.shape,
            #       beam_scores[:, None].expand_as(next_token_scores))
            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(
                next_token_scores)
            # [batch_size, beam_size * vocab_size]
            reshape_size = next_token_scores.shape[0] // batch_size
            # TODO: 得到beam_size个预测结果后，将beam_size组预测结果堆在统一维度
            # 即维度由原本的 [batch_size * beam_size, vocab_size] -> [batch_size, beam_size * vocab_size]
            next_token_scores = rearrange(
                next_token_scores,
                "(b m) v -> b (m v)",
                m=reshape_size,
            )

            next_token_scores, next_tokens = torch.topk(next_token_scores,
                                                        2 * beam_size,
                                                        dim=1)
            # print("top K prob shape & value", next_tokens.shape, next_tokens)
            # print("Top K next_token1: ", next_tokens)
            # TODO: 将vocab_size * beam_size对应到各自的类别
            next_indices = next_tokens // vocab_size
            # print("Top K next_indices: ", next_indices)
            next_tokens = next_tokens % vocab_size
            # print("Top K next_token2: ", next_tokens)

            if cur_len == 1:
                # TODO: 如果是第一个推理则将所有数据扩展为beam_size倍
                input_ids = repeat(input_ids, "b l -> (b m) l", m=beam_size)
                for i in range(len(src)):
                    src[i] = repeat(src[i], "b ... -> (b m) ...", m=beam_size)
                    src_mask[i] = repeat(src_mask[i],
                                         "b ... -> (b m) ...",
                                         m=beam_size)
            # print(f"cur_len : {cur_len} input_ids shape & value: ", input_ids.shape, input_ids)
            beam_scores, beam_next_tokens, beam_idx = beam_scorer.process(
                input_ids=input_ids,
                next_scores=next_token_scores,
                next_tokens=next_tokens,
                next_indices=next_indices)
            # print("beam_next_structs shape & value: ", beam_next_structs.shape, beam_next_structs)
            input_ids = torch.cat(
                (input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)),
                dim=-1)
            cur_len += 1
        # print("last struct_out: ", struct_out.shape)
        return beam_scorer.finalize(input_ids, beam_scores)

    def _rate(
            self,
            src: List[FloatTensor],
            src_mask: List[LongTensor],
            tgt: LongTensor,
            out: LongTensor,
            alpha: float,
            temperature: float,
    ) -> FloatTensor:
        """rate tgt and output

        Parameters
        ----------
        src : List[FloatTensor]
            [b * beam_size, t, d]
        src_mask : List[LongTensor]
            [b * beam_size, t]
        tgt : LongTensor
            [b * beam_size, l]
        out : LongTensor
            [b * beam_size, l]
        alpha : float
        temperature : float

        Returns
        -------
        FloatTensor
            [b * beam_size]
        """
        b = tgt.shape[0]
        word_out, tree_attn, word_attn = self.transform(src, src_mask, tgt, tgt)
        # print("struct_out : ", struct_out.shape, struct_out)
        out_hat = word_out / temperature

        loss = ce_loss(out_hat, out, reduction="none")
        loss = rearrange(loss, "(b l) -> b l", b=b)

        mask = tgt == vocab.PAD_IDX
        penalty = (~mask).sum(dim=1) ** alpha
        loss = -torch.sum(loss, dim=1) / penalty

        return loss, tree_attn, word_attn


def my_convert(gtd_list):
    struct_stack = []
    result_str = []
    # struct_list = ["above", "below", "sub", "sup", "s-sub", "s-sup", "l-sup", "inside", "right", "sum above", "sum below"]
    struct_list = ["above", "below", "sub", "sup", "l-sup", "inside", "right", "sum above", "sum below"]
    for i in range(len(gtd_list)):
        word_now = vocab.idx2word[gtd_list[i]]
        # word_now = gtd_list[i]
        # print("convert: ", word_now)
        if word_now not in struct_list:
            result_str += [word_now]
        else:
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
                if i > 0 and vocab.idx2word[gtd_list[i - 1]] in ["\\sum", "\\lim", "\\int"]:
                    result_str += ['_', '{']
                    struct_stack.append("sum below")
                else:
                    if len(struct_stack) != 0:
                        while len(struct_stack):
                            struct = struct_stack.pop()
                            if struct == "above":
                                result_str += ['}']
                                break
                            elif struct == "l-sup":
                                result_str += [']']
                            else:
                                result_str += ['}']
                        result_str += ['{']
                    else:
                        result_str += ['{']
                    struct_stack.append(word_now)
            elif word_now == "above":
                if i > 0 and vocab.idx2word[gtd_list[i - 1]] == "\\frac":
                    result_str += ['{']
                    struct_stack.append("above")
                else:
                    while len(struct_stack):
                        struct = struct_stack.pop()
                        if struct == "sum below":
                            result_str += ['}']
                            break
                        elif struct == "l-sup":
                            result_str += [']']
                        else:
                            result_str += ['}']
                        if len(struct_stack):
                            result_str += ['{']
                    result_str += ['^', '{']
                    struct_stack.append("sum above")
            else:
                if word_now == "inside":
                    if len(struct_stack) and struct_stack[-1] == "l-sup":
                        struct_stack.pop()
                        result_str += [']']
                    result_str += ['{']
                elif word_now == "sup":
                    if len(struct_stack):
                        if struct_stack[-1] == "sub":
                            struct_stack.pop()
                            result_str += ['}']
                    result_str += ['^', '{']
                elif word_now == "sub":
                    if len(struct_stack):
                        if struct_stack[-1] == "sup":
                            struct_stack.pop()
                            result_str += ['}']
                    result_str += ['_', '{']
                # elif word_now == "s-sup":
                #     if len(struct_stack):
                #         if struct_stack[-1] == "sub":
                #             struct_stack.pop()
                #             result_str += ['}']
                #     result_str += ['^', '{']
                # elif word_now == "s-sub":
                #     if len(struct_stack):
                #         if struct_stack[-1] == "sup":
                #             struct_stack.pop()
                #             result_str += ['}']
                #     result_str += ['_', '{']
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


# def my_convert(gtd_list):
#     struct_stack = []
#     result_str = []
#     struct_list = ["above", "below", "sub", "sup", "l-sup", "inside", "right"]
#     for i in range(len(gtd_list)):
#         word_now = vocab.idx2word[gtd_list[i]]
#         # print("convert: ", word_now)
#         if word_now not in struct_list:
#             result_str += [word_now]
#         else:
#             if word_now == "right":
#                 if len(struct_stack) != 0:
#                     struct = struct_stack.pop()
#                     if struct == "l-sup":
#                         result_str += [']']
#                     else:
#                         result_str += ['}']
#                 else:
#                     continue
#             elif word_now == "below":
#                 if i > 0 and gtd_list[i-1] == "\\sum":
#                     result_str += ['_', '{']
#                 else:
#                     if len(struct_stack) != 0:
#                         while len(struct_stack):
#                             struct = struct_stack.pop()
#                             if struct == "above":
#                                 result_str += ['}']
#                                 break
#                             elif struct == "l-sup":
#                                 result_str += [']']
#                             else:
#                                 result_str += ['}']
#                         result_str += ['{']
#                     else:
#                         result_str += ['{']
#                 struct_stack.append(word_now)
#             else:
#                 if word_now == "sup":
#                     if len(struct_stack) != 0:
#                         if struct_stack[-1] == "sub":
#                             struct_stack.pop()
#                             result_str += ['}']
#                     if result_str[-1] in ["\\sum", "\\lim", "\\int"]:
#                         # result_str += ['limits', '^', '{']
#                         result_str += ['^', '{']
#                     else:
#                         result_str += ['^', '{']
#                 elif word_now == "sub":
#                     if len(struct_stack) != 0:
#                         if struct_stack[-1] == "sup":
#                             struct_stack.pop()
#                             result_str += ['}']
#                     if result_str[-1] in ["\\sum", "\\lim", "\\int"]:
#                         # result_str += ['\\limits', '_', '{']
#                         result_str += ['_', '{']
#                     else:
#                         result_str += ['_', '{']
#                 elif word_now == "l-sup":
#                     result_str += ['[']
#                 else:
#                     result_str += ['{']
#                 struct_stack.append(word_now)
#     while len(struct_stack) != 0:
#         struct = struct_stack.pop()
#         if struct == "l-sup":
#             result_str += [']']
#         else:
#             result_str += ['}']
#     return " ".join(result_str)



def convert(nodeid, gtd_list):
    isparent = False
    child_list = []
    for i in range(len(gtd_list)):
        if gtd_list[i][2] == nodeid:
            isparent = True
            child_list.append([gtd_list[i][0], gtd_list[i][1], gtd_list[i][3]])
    if not isparent:
        return [gtd_list[nodeid][0]]
    else:
        if gtd_list[nodeid][0] == '\\frac':
            return_string = [gtd_list[nodeid][0]]
            for i in range(len(child_list)):
                if child_list[i][2] == 'Above':
                    return_string += ['{'] + convert(child_list[i][1],
                                                     gtd_list) + ['}']
            for i in range(len(child_list)):
                if child_list[i][2] == 'Below':
                    return_string += ['{'] + convert(child_list[i][1],
                                                     gtd_list) + ['}']
            for i in range(len(child_list)):
                if child_list[i][2] == 'Right':
                    return_string += convert(child_list[i][1], gtd_list)
            for i in range(len(child_list)):
                if child_list[i][2] not in ['Right', 'Above', 'Below']:
                    return_string += ['illegal']
        else:
            return_string = [gtd_list[nodeid][0]]
            for i in range(len(child_list)):
                if child_list[i][2] in ['l_sup']:
                    return_string += ['['] + convert(child_list[i][1],
                                                     gtd_list) + [']']
            for i in range(len(child_list)):
                if child_list[i][2] == 'Inside':
                    return_string += ['{'] + convert(child_list[i][1],
                                                     gtd_list) + ['}']
            for i in range(len(child_list)):
                if child_list[i][2] in ['Sub', 'Below']:
                    return_string += ['_', '{'] + convert(
                        child_list[i][1], gtd_list) + ['}']
            for i in range(len(child_list)):
                if child_list[i][2] in ['Sup', 'Above']:
                    return_string += ['^', '{'] + convert(
                        child_list[i][1], gtd_list) + ['}']
            for i in range(len(child_list)):
                if child_list[i][2] in ['Right']:
                    return_string += convert(child_list[i][1], gtd_list)
        return return_string


def prediction(word_list, struct_props):
    struct_dict = [109, 110, 111, 112, 113, 114, 115]
    prediction = ''
    struct_list = []
    right_brace = 0
    cid, pid = 0, 0
    p_re = 'Start'
    word = torch.LongTensor([1])
    result = [['<s>', 0, -1, 'root']]
    currentIndex = 0
    while len(prediction) < 400:
        p_word = word
        # _, word = word_prob.max(1)
        word = word_list[currentIndex]
        # print("prediction: ", word)
        if word and word != 3:
            cid += 1
            p_id = cid
            result.append([vocab.idx2word[word], cid, pid, p_re])
            prediction = prediction + vocab.idx2word[word] + ' '
        #
        # 当预测文字为结构符
        if word == 3:
            # struct_prob = self.struct_convert(word_out_state)
            structs = struct_props[currentIndex]
            # print(structs)
            # structs = torch.sigmoid(struct_prob)
            # print("San_decoder struct shape: ", structs.shape)
            # 逆向遍历structs概率
            # print("struct infer: ", structs)
            for num in range(len(structs) - 1, -1, -1):
                # print("struct infer: ", structs[num])
                if structs[num] > 0.5:
                    struct_list.append((struct_dict[num], p_word, p_id))
                    # print("San_decoder range shape: ", (self.struct_dict[num], hidden.shape, p_word, p_id, word_alpha_sum.shape))

            # struct节点必须包含位置信息
            if len(struct_list) == 0:
                break
            word, p_word, pid = struct_list.pop()
            if word == 111 or (word == 110 and p_word == 69):
                prediction = prediction + '_ { '
                p_re = 'Sub'
                right_brace += 1
            elif word == 112 or (word == 109 and p_word == 69):
                p_re = 'Sup'
                prediction = prediction + '^ { '
                right_brace += 1
            elif word == 109 and p_word == 15:
                p_re = 'Above'
                prediction = prediction + '{ '
                right_brace += 1
            elif word == 110 and p_word == 15:
                p_re = 'Below'
                prediction = prediction + '{ '
                right_brace += 1
            elif word == 113:
                p_re = 'l_sup'
                prediction = prediction + '[ '
            elif word == 114:
                p_re = 'Inside'
                prediction = prediction + '{ '
                right_brace += 1
        # 结束符
        elif word == 1:
            # print("San_decoder word == 0 : ", word)
            if len(struct_list) == 0:
                if right_brace != 0:
                    for brach in range(right_brace):
                        prediction = prediction + '} '
                break
            word, p_word, pid = struct_list.pop()
            if word == 114:
                prediction = prediction + '] { '
                right_brace += 1
                p_re = 'Inside'
            elif word == 111 or (word == 110 and p_word == 69):
                p_re = 'Sub'
                prediction += '} '
                right_brace -= 1
                if right_brace != 0:
                    for num in range(right_brace):
                        prediction += '} '
                        right_brace -= 1
                prediction = prediction + '_ { '
                right_brace += 1
            elif word == 112 or (word == 109 and p_word == 69):
                p_re = 'Sup'
                prediction += '} '
                right_brace -= 1
                if right_brace != 0:
                    for num in range(right_brace):
                        prediction += '} '
                        right_brace -= 1
                prediction = prediction + '^ { '
                right_brace += 1
            elif word == 109 and p_word == 15:
                p_re = 'Above'
                prediction += '} '
                right_brace -= 1
                if right_brace != 0:
                    for num in range(right_brace):
                        prediction += '} '
                        right_brace -= 1
                prediction = prediction + '{ '
                right_brace += 1
            elif word == 110 and p_word == 15:
                p_re = 'Below'
                prediction += '} '
                right_brace -= 1
                if right_brace != 0:
                    for num in range(right_brace):
                        prediction += '} '
                        right_brace -= 1
                prediction = prediction + '{ '
                right_brace += 1
            elif word == 113:
                p_re = 'l_sup'
                prediction = prediction + '[ '
            elif word == 114:
                p_re = 'Inside'
                prediction = prediction + '] { '
                right_brace += 1
            elif word == 115:
                p_re = 'Right'
                prediction = prediction + '} '
                right_brace -= 1
        # 处理父节点
        else:
            # print("San_decoder word != 0 : ", self.params['words'].words_index_dict[word])
            p_re = 'Right'
            pid = cid
            # word_embedding = self.embedding(word)
            # parent_hidden = hidden.clone()
        currentIndex += 1
    # print(prediction)
    return result
