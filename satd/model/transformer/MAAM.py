import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch import Tensor
from torch.nn.modules.batchnorm import BatchNorm1d


class MaskBatchNorm2d(nn.Module):

    def __init__(self, num_features: int):
        super().__init__()
        self.bn = BatchNorm1d(num_features)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            [b, d, h, w]
        mask : Tensor
            [b, 1, h, w]

        Returns
        -------
        Tensor
            [b, d, h, w]
        """
        # [b * l, n_head, h, w] -> [b * l, h, w, n_head]
        x = rearrange(x, "b d h w -> b h w d")  # [b * l, h, w, n_head]
        mask = mask.squeeze(1)  # [b * l, h, w]

        # mask中为True表示被mask
        # 因此not_mask中为True表示没被mask
        not_mask = ~mask

        # [b * l * (h' * w'), n_head]
        # 其中h' * w' 表示图像特征中没有被mask掉的特征数
        flat_x = x[not_mask, :]  # [b * l * (h' * w'), n_head]
        flat_x = self.bn(flat_x)  # 对没有被mask的数据，以n_head为单位做bn
        x[not_mask, :] = flat_x  # 将bn完成的数据放回原tensor的对应位置

        x = rearrange(x, "b h w d -> b d h w")  # [b*l, n_head, h, w]

        return x


class MultiscaleAttentionAggregationModule(nn.Module):

    def __init__(self, nhead: int, dc: int, cross_coverage: bool,
                 self_coverage: bool):
        """注意力精炼模块

        Args:
            nhead (int): 注意力头数
            dc (int): 卷积层输出通道数
            cross_coverage (bool): 层间注意力
            self_coverage (bool): 层内注意力
        """
        super().__init__()
        assert cross_coverage or self_coverage
        self.nhead = nhead
        self.cross_coverage = cross_coverage
        self.self_coverage = self_coverage

        if cross_coverage and self_coverage:
            in_chs = 2 * nhead
        else:
            in_chs = nhead
        # self.conv = nn.Conv2d(in_chs, dc, kernel_size=3, padding=1)
        self.conv = nn.Conv2d(in_chs, dc, kernel_size=5, padding=2)
        self.act = nn.ReLU(inplace=True)

        self.proj = nn.Conv2d(dc, nhead, kernel_size=1, bias=False)

        # self.conv2 = nn.Conv2d(in_chs, dc, kernel_size=13, padding=6)
        # self.conv2 = nn.Conv2d(in_chs, dc, kernel_size=5, padding=2)
        # self.conv2 = nn.Conv2d(in_chs, dc, kernel_size=7, padding=3)
        # self.conv2 = nn.Conv2d(in_chs, dc, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(in_chs, dc, kernel_size=11, padding=5)
        self.act2 = nn.ReLU(inplace=True)

        self.proj2 = nn.Conv2d(dc, nhead, kernel_size=1, bias=False)
        self.post_norm = MaskBatchNorm2d(nhead)

    def forward(self, prev_attn: Tensor, key_padding_mask: Tensor, h: int,
                curr_attn: Tensor) -> Tensor:
        """
        Parameters
        ----------
        prev_attn : Tensor
            [(b * nhead), t, l]
        key_padding_mask : Tensor
            [b, l]
        h : int

        Returns
        -------
        Tensor
            [(b * nhead), t, l]
        """
        t = curr_attn.shape[1]  # t = query_len

        # [2b, src_l] -> [2b * l, 1, h, w]
        # l = query_l
        # h * w = src_l
        res = curr_attn
        mask = repeat(key_padding_mask, "b (h w) -> (b t) () h w", h=h, t=t)  # [2b * l, 1, h, w]

        curr_attn = rearrange(curr_attn, "(b n) t l -> b n t l", n=self.nhead)  # [2b, num_head, l, src_l]
        prev_attn = rearrange(prev_attn, "(b n) t l -> b n t l", n=self.nhead)  # [2b, num_head, l, src_l]

        attns = []
        if self.cross_coverage:
            attns.append(prev_attn)
        if self.self_coverage:
            attns.append(curr_attn)
        attns = torch.cat(attns, dim=1)  # [b, 2n, l, src_l]

        # 在 t 轴上进行前缀和计算(即attns[:,:,1,:] = attns[:, :, 0, :] + attns[:, :, 1, :])，然后减去原来的值
        # 即 t 位之前的权重之和
        attns = attns.cumsum(dim=2) - attns  # [b, 2n, l, src_l]
        attns = rearrange(attns, "b n t (h w) -> (b t) n h w", h=h)  # [b * l, 2n, h, w]


        cov = self.conv(attns)  # [b * l, dc, h, w]
        cov = self.act(cov)

        cov = cov.masked_fill(mask, 0.0)  # [b * l, dc, h, w]masked with 0.0
        # 使用1*1卷积作为线性层, 将维度复原为n_head, 即融合之前和当前的注意力权重
        cov = self.proj(cov)  # [b * l, n_head, h, w]

        cov2 = self.conv2(attns)
        cov2 = self.act(cov2)
        cov2 = cov2.masked_fill(mask, 0.0)
        cov2 = self.proj2(cov2)

        cov = cov + cov2

        cov = self.post_norm(cov, mask)  # [b * l, n_head, h, w]

        cov = rearrange(cov, "(b t) n h w -> (b n) t (h w)", t=t)  # [b * n_head, l, src_l]

        return cov
