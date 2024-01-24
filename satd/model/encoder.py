import math
from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange
from torch import FloatTensor, LongTensor
from torch.nn import init

from .pos_enc import ImgPosEnc


# 通道注意力机制Channel Attention
class ChannelAttention(nn.Module):
    def __init__(self, in_channel, ratio=16):
        super(ChannelAttention, self).__init__()
        # 平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 最大池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # MLP
        self.fc_in = nn.Conv2d(in_channel, in_channel // ratio, 1, bias=False)
        self.act1 = nn.ReLU()
        self.fc_out = nn.Conv2d(in_channel // ratio, in_channel, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.sigmoid(self.fc_out(self.act1(self.fc_in(self.avg_pool(x)))))
        max_out = self.sigmoid(self.fc_out(self.act1(self.fc_in(self.max_pool(x)))))

        out = avg_out + max_out

        return out


# 空间注意力机制Spatial Attention
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # 核大小必须为3或7
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        self.padding = 3 if kernel_size == 7 else 1

        self.cov1 = nn.Conv2d(2, 1, kernel_size, padding=self.padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 平均池化
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # 最大池化
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # concat
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.cov1(x)
        return self.sigmoid(x)


class CMBABlock(nn.Module):
    def __init__(self, in_channel, ratio=16, kernel_size=7):
        super(CMBABlock, self).__init__()
        self.channel_attention = ChannelAttention(in_channel=in_channel, ratio=ratio)
        self.spatial_attention = SpatialAttention(kernel_size=kernel_size)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        res = x
        out = x * self.channel_attention(x)
        out = out * self.spatial_attention(out)
        return out + res


# DenseNet-B
# RestNet中提出的bottleneck块
class _Bottleneck(nn.Module):

    def __init__(self, n_channels: int, growth_rate: int, use_dropout: bool, use_vision_attention: bool = False):
        super(_Bottleneck, self).__init__()
        self.use_vision_attention = use_vision_attention
        interChannels = 4 * growth_rate
        self.bn1 = nn.BatchNorm2d(interChannels)
        self.conv1 = nn.Conv2d(n_channels,
                               interChannels,
                               kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(growth_rate)
        if self.use_vision_attention:
            self.vision_attention1 = CMBABlock(interChannels)
        self.conv2 = nn.Conv2d(interChannels,
                               growth_rate,
                               kernel_size=3,
                               padding=1,
                               bias=False)
        if self.use_vision_attention:
            self.vision_attention2 = CMBABlock(growth_rate)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # 1*1卷积增加通道数 -> BN -> ReLu -> 3*3卷积提取特征下采样 -> bn -> ReLu -> 残差连接
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        if self.use_vision_attention:
            out = self.vision_attention1(out)
        if self.use_dropout:
            out = self.dropout(out)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        if self.use_vision_attention:
            out = self.vision_attention2(out)
        if self.use_dropout:
            out = self.dropout(out)
        out = torch.cat((x, out), 1)
        return out


# single layer
class _SingleLayer(nn.Module):

    def __init__(self, n_channels: int, growth_rate: int, use_dropout: bool, use_vision_attention: bool = False):
        super(_SingleLayer, self).__init__()
        self.use_vision_attention = use_vision_attention
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.conv1 = nn.Conv2d(n_channels,
                               growth_rate,
                               kernel_size=3,
                               padding=1,
                               bias=False)
        if use_vision_attention:
            self.vision_attention = CMBABlock(growth_rate)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # ReLu -> 3*3卷积下采样 -> 残差连接
        out = self.conv1(F.relu(x, inplace=True))
        if self.use_vision_attention:
            out = self.vision_attention(out)
        if self.use_dropout:
            out = self.dropout(out)
        out = torch.cat((x, out), 1)
        return out


# transition layer
class _Transition(nn.Module):

    def __init__(self, n_channels: int, n_out_channels: int,
                 use_dropout: bool, use_vision_attention: bool = False):
        super(_Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(n_out_channels)
        self.use_vision_attention = use_vision_attention
        self.conv1 = nn.Conv2d(n_channels,
                               n_out_channels,
                               kernel_size=1,
                               bias=False)
        if use_vision_attention:
            self.vision_attention = CMBABlock(n_out_channels)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # 1*1卷积改变通道数 -> BN -> ReLu -> 2*2 avg_pooling
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        if self.use_vision_attention:
            out = self.vision_attention(out)
        if self.use_dropout:
            out = self.dropout(out)
        out = F.avg_pool2d(out, 2, ceil_mode=True)
        return out


class DenseNet(nn.Module):

    def __init__(
            self,
            growth_rate: int,
            num_layers: int,
            reduction: float = 0.5,
            bottleneck: bool = True,
            use_dropout: bool = True,
            use_vision_attention: bool = False,
    ):
        super(DenseNet, self).__init__()
        # dense块数
        n_dense_blocks = num_layers
        # 输出通道数
        n_channels = 2 * growth_rate
        self.conv1 = nn.Conv2d(1,
                               n_channels,
                               kernel_size=7,
                               padding=3,
                               stride=2,
                               bias=False)
        self.norm1 = nn.BatchNorm2d(n_channels)
        self.dense1 = self._make_dense(n_channels, growth_rate, n_dense_blocks,
                                       bottleneck, use_dropout, use_vision_attention)
        n_channels += n_dense_blocks * growth_rate
        n_out_channels = int(math.floor(n_channels * reduction))
        self.trans1 = _Transition(n_channels, n_out_channels, use_dropout, use_vision_attention)

        n_channels = n_out_channels
        self.dense2 = self._make_dense(n_channels, growth_rate, n_dense_blocks,
                                       bottleneck, use_dropout, use_vision_attention)
        n_channels += n_dense_blocks * growth_rate
        n_out_channels = int(math.floor(n_channels * reduction))
        self.trans2 = _Transition(n_channels, n_out_channels, use_dropout, use_vision_attention)

        n_channels = n_out_channels
        self.dense3 = self._make_dense(n_channels, growth_rate, n_dense_blocks,
                                       bottleneck, use_dropout, use_vision_attention)

        self.out_channels = n_channels + n_dense_blocks * growth_rate
        self.post_norm = nn.BatchNorm2d(self.out_channels)

    @staticmethod
    def _make_dense(n_channels, growth_rate, n_dense_blocks, bottleneck,
                    use_dropout, use_vision_attention):
        layers = []
        for _ in range(int(n_dense_blocks)):
            if bottleneck:
                layers.append(_Bottleneck(n_channels, growth_rate,
                                          use_dropout, use_vision_attention))
            else:
                layers.append(
                    _SingleLayer(n_channels, growth_rate, use_dropout, use_vision_attention))
            n_channels += growth_rate
        return nn.Sequential(*layers)

    def forward(self, x, x_mask):
        out = self.conv1(x)
        out = self.norm1(out)
        # out_mask = x_mask[:, 0::2, 0::2]
        out = F.relu(out, inplace=True)
        out = F.max_pool2d(out, 2, ceil_mode=True)
        # print("Before Dense1 feature shape: ", out.shape)
        # out_mask = out_mask[:, 0::2, 0::2]
        out = self.dense1(out)
        out = self.trans1(out)
        # out_mask = out_mask[:, 0::2, 0::2]
        # print("Before Dense2 feature shape: ", out.shape)
        out = self.dense2(out)
        out = self.trans2(out)
        # print("Before Dense3 feature shape: ", out.shape)
        out_mask = x_mask[:, 0::16, 0::16]
        out = self.dense3(out)
        out = self.post_norm(out)
        # print("After Dense3 feature shape: ", out.shape)
        return out, out_mask


class Encoder(pl.LightningModule):

    def __init__(self, d_model: int, growth_rate: int, num_layers: int, use_vision_attention: bool = False):
        super().__init__()

        self.model = DenseNet(growth_rate=growth_rate, num_layers=num_layers, use_vision_attention=use_vision_attention)

        self.feature_proj = nn.Conv2d(self.model.out_channels,
                                      d_model,
                                      kernel_size=1)

        self.pos_enc_2d = ImgPosEnc(d_model, normalize=True)

        self.norm = nn.LayerNorm(d_model)

    def forward(self, img: FloatTensor,
                img_mask: LongTensor) -> Tuple[FloatTensor, LongTensor]:
        """encode image to feature

        Parameters
        ----------
        img : FloatTensor
            [b, 1, h', w']
        img_mask: LongTensor
            [b, h', w']

        Returns
        -------
        Tuple[FloatTensor, LongTensor]
            [b, h, w, d], [b, h, w]
        """
        # extract feature
        feature, mask = self.model(img, img_mask)
        feature = self.feature_proj(feature)

        # proj
        feature = rearrange(feature, "b d h w -> b h w d")

        # positional encoding
        feature = self.pos_enc_2d(feature, mask)
        # fig, axes = plt.subplots(1, 5)
        feature = self.norm(feature)
        # for i in range(len(feature[0])):
        #     if i == 5:
        #         break
        #     axes[0][i].imshow(feature[0][:][:][i].cpu().numpy())
        #     axes[0][i].set_title("With Position Encoding")
        #     axes[0][i].axis("off")
        # plt.show()

        # flat to 1-D
        return feature, mask

    # def forward(self, img: FloatTensor, img_mask: LongTensor):
    #     """encode image to feature

    #     Parameters
    #     ----------
    #     img : FloatTensor
    #         [b, 1, h', w']
    #     img_mask: LongTensor
    #         [b, h', w']

    #     Returns
    #     -------
    #     Tuple[FloatTensor, LongTensor]
    #         [b, h, w, d], [b, h, w]
    #     """
    #     # extract feature
    #     feature, mask = self.model(img, img_mask)
    #     feature = self.feature_proj(feature)

    #     # proj
    #     feature = rearrange(feature, "b d h w -> b h w d")

    #     # positional encoding
    #     feature = self.pos_enc_2d(feature, mask)
    #     feature = self.norm(feature)

    #     # flat to 1-D
    #     return feature
