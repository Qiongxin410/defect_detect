'''
本文件建立了编码器的主干网络, 参考ConvNeXt v2网络
对外接口函数为feature_image_model()
参数:
    in_chans: 输入图像的通道数, 默认为3
    depths: 每个阶段的block数, 默认为[3, 3, 27, 3]
    dims: 每个阶段特征图的维度, 默认为[128, 256, 512, 1024]
模型：
    输入: [B, C, H, W]
    初始降采样: [B, dim[0], H // 4, W // 4]
    阶段1: [B, dim[0], H // 4, W // 4]
    降采样1: [B, dim[1], H // 8, W // 8]
    阶段2: [B, dim[1], H // 8, W // 8]
    降采样2: [B, dim[2], H // 16, W // 16]
    阶段3: [B, dim[2], H // 16, W // 16]
    降采样3: [B, dim[3], H // 32, W // 32]
    阶段4: [B, dim[3], H // 32, W // 32]
例:
    输入torch.tensor(16, 3, 1024, 1024)
    输出torch.tensor(16, 1024, 32, 32)
待完善: 
    对图像序列(视频)的处理
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_


class LayerNorm(nn.Module):
    ''' Layer Normalization
    Args: 
        normalized_shape: Feature dimension.
        eps: Minimum stride.
        data_format: channels_first for [B, C, H, W] and channels_last for [B, H, W, C]
    '''
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class GRN(nn.Module):
    ''' Ground Rseponse Normalization
    Args: 
        dim: Feature dimension.
    '''
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


# 本网络参考ConvNeXt v2
class Block(nn.Module):
    """ Basic Block.
    Args:
        dim (int): Number of input channels.
    """
    def __init__(self, dim):
        super(Block, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        input = x
        x = self.dwconv(x) # [B, dim, H, W] -> [B, dim, H, W]
        x = x.permute(0, 2, 3, 1) # [B, dim, H, W] -> [B, H, W, dim]
        x = self.norm(x) # [B, H, W, dim] -> [B, H, W, dim]
        x = self.pwconv1(x) # [B, H, W, dim] -> [B, H, W, 4 * dim]
        x = self.act(x) # [B, H, W, 4 * dim] -> [B, H, W, 4 * dim]
        x = self.grn(x) # [B, H, W, 4 * dim] -> [B, H, W, 4 * dim]
        x = self.pwconv2(x) # [B, H, W, 4 * dim] -> [B, H, W, dim]
        x = x.permute(0, 3, 1, 2) # [B, dim, H, W] -> [B, dim, H, W]

        x = input + x # [B, dim, H, W] -> [B, dim, H, W]
        return x

class ConvNeXtV2(nn.Module):
    """ ConvNeXt V2
    Args:
        in_chans (int): Number of input image channels. Default: 3
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 27, 3]
        dims (int): Feature dimension at each stage. Default: [128, 256, 512, 1024]
    """
    def __init__(self, in_chans=3, depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024]):
        super(ConvNeXtV2, self).__init__()
        self.depths = depths
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        '''
        [B, dims[1], H, W] -> [B, dims[2], H // 4, W // 4]
        '''
        for i in range(3):
            '''
            [B, dims[0], H // 4, W // 4] -> [B, dims[1], H // 8, W // 8]
            [B, dims[1], H // 8, W // 8] -> [B, dims[2], H // 16, W // 16]
            [B, dims[2], H // 16, W // 16] -> [B, dims[3], H // 32, W // 32]
            '''
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        for i in range(4):
            '''
            [B, dim[0], H // 4, W // 4] -> [B, dim[0], H // 4, W // 4]
            [B, dim[1], H // 8, W // 8] -> [B, dim[1], H // 8, W // 8]
            [B, dim[2], H // 16, W // 16] -> [B, dim[2], H // 16, W // 16]
            [B, dim[3], H // 32, W // 32] -> [B, dim[3], H // 32, W // 32]
            '''
            stage = nn.Sequential(Block(dim=dims[i]))
            self.stages.append(stage)

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        return x


def feature_image_model(in_chans=3, depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024]):
    model = ConvNeXtV2(in_chans=in_chans, depths=depths, dims=dims)

    return model


if __name__ == '__main__':
    x = torch.rand(16, 3,1024, 1024)
    model = feature_image_model()
    output = model(x)
    print(output.shape)