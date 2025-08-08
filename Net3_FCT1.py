import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from block11_25_315 import Block

from timm.models.layers import DropPath, trunc_normal_, to_2tuple
from Separable_convolution import S_conv
from functools import partial
from PSA import PSA_p

nonlinearity = partial(F.relu, inplace=True)


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

def pool_global(kernel_size):
    return nn.AvgPool2d(kernel_size=kernel_size, stride=kernel_size, padding=0)
class DeBlock(nn.Module):
    def __init__(self, in_channels):
        super(DeBlock, self).__init__()

        self.conv1 = S_conv(in_channels, in_channels)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(in_channels)
        # self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = S_conv(in_channels, in_channels)
        # self.bn2 = nn.BatchNorm2d(in_channels)
        self.relu2 = nn.ReLU(inplace=True)
        # self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)
        x1 = x1 + x
        x1 = x1.reshape(B, C, -1).permute(0, 2, 1)
        return x1


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, seq_length):
        super(LearnedPositionalEncoding, self).__init__()

        self.position_embeddings = nn.Parameter(torch.zeros(1, seq_length, embedding_dim))  # 8x

    def forward(self, x):
        position_embeddings = self.position_embeddings
        return x + position_embeddings


class SelfAttention(nn.Module):
    def __init__(self, dim,window_size, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.1):
        super().__init__()
        self.dim = dim
        self.group_size = window_size  # Wh, Ww
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(p=0.1)

        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x ,y):
        a = x
        b = y
        B, N, C = x.shape
        qkv1 = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv2 = self.qkv(y).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv1[0], qkv1[1], qkv1[2]# make torchscript happy (cannot use tensor as tuple)
        q1, k1, v1 = qkv2[0], qkv2[1],qkv2[2]  # ma
        #
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        #
        q1 = q1 * self.scale
        attn1 = (q1 @ k1.transpose(-2, -1))
        attn1 = self.softmax(attn1)
        attn1 = self.attn_drop(attn1)

        x = (attn @ v1).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x + a
        #
        y = (attn1 @ v).transpose(1, 2).reshape(B, N, C)
        y = self.proj(y)
        y = self.proj_drop(y)
        y = y + b
        return x,y


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            # nn.GELU(),改了
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout_rate),
        )


    def forward(self,  x, y):

        x, y = self.net(x), self.net(y)
        return  x, y


class Conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Conv_block, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out // 2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out // 2)
        self.conv2 = nn.Conv2d(ch_out // 2, ch_out, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1)

    def forward(self, x):
        residual = self.conv1x1(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = residual + out
        out = self.relu(out)
        return out


class Trans_CNN_DeBlock(nn.Module):
    def __init__(self, in_channels,resolution,dropout_rate=0.1,window_size=16):

        super(Trans_CNN_DeBlock, self).__init__()

        self.conv1 = S_conv(in_channels, in_channels)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.conv2 = S_conv(in_channels, in_channels)
        self.relu2 = nn.ReLU()
        #
        self.linear_encoding = nn.Linear(in_channels, in_channels//2)
        self.linear_encoding_de = nn.Linear(in_channels//2, in_channels)
        self.position_encoding = LearnedPositionalEncoding(in_channels//2 ,resolution * resolution)
        self.norm = nn.LayerNorm(in_channels//2)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.satt = SelfAttention(in_channels//2,window_size)
        self.ffn =FeedForward(in_channels//2,in_channels//2,dropout_rate)
        self.window_size = window_size
        self.conv_b = Conv_block(in_channels, in_channels)
    def forward(self, x_C,x_T):
        B, N, C = x_C.shape
        H = W = int(N ** 0.5)
        x_C = x_C.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x_Cres =x_C
        x1 = self.conv1(x_C)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = x1 + x_C
        x_T = x_T.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x_Tres = x_T
        x2 = self.conv1(x_T)
        x2 = self.bn1(x_T)
        x2 = self.relu1(x_T)
        x2 = x2 + x_T
        ###新增
        x_ = x1.permute(0, 2, 3, 1).contiguous()#B, H, W, C
        y_ = x2.permute(0, 2, 3, 1).contiguous()
        x = x_.view(x_.size(0), x_.size(2) * x_.size(1), -1)
        y = y_.view(y_.size(0), y_.size(2) * y_.size(1), -1)

        x_poi= self.position_encoding(self.linear_encoding(x))
        y_poi = self.position_encoding(self.linear_encoding(y))

        ##改为窗口减少计算压力
        wx_poi = x_poi.reshape(B, H, W, -1)
        wy_poi = y_poi.reshape(B, H, W, -1)
        G = self.window_size ##G决定了注意力的权重矩阵的大小（G²*G²）
        wx_poi = wx_poi.reshape(B, H // G, G, W // G, G, C//2).permute(0, 1, 3, 2, 4, 5)
        wx_poi = wx_poi.reshape(B * H * W // G ** 2, G ** 2, C//2)
        wy_poi = wy_poi.reshape(B, H // G, G, W // G, G, C//2).permute(0, 1, 3, 2, 4, 5)
        wy_poi = wy_poi.reshape(B * H * W // G ** 2, G ** 2, C//2)
        wx_poi =self.norm(wx_poi)
        wy_poi =self.norm(wy_poi)
        x,y= self.satt(wx_poi,wy_poi)[0],self.satt(wx_poi,wy_poi)[1]   #1024 16 48
        x = x.reshape(B, H // G, W // G, G, G,C//2)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, C//2)
        x = x.contiguous().view(B, H * W, C//2)
        y = y.reshape(B, H // G, W // G, G, G, C//2)
        y = y.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, C//2)
        y = y.contiguous().view(B, H * W, C//2)

        y_ffn = self.norm(y)
        x_ffn = self.norm(x)
        x_ffn, y_ffn= self.ffn(x_ffn,y_ffn)

        x, y =x+ x_ffn,y+y_ffn    #残差链接
        x = self.linear_encoding_de(x).permute(0, 2, 1).contiguous()
        x = x.view(x.size(0), x.size(1), H, W)
        y = self.linear_encoding_de(y).permute(0, 2, 1).contiguous()
        y = y.view(y.size(0), y.size(1), H, W)
        # # 增3*3卷积块
        # x = self.conv_b(x)
        # x = x+ x_Cres
        # y = self.conv_b(y)
        # y = y + x_Tres
        output = to_3d(y+x)
        return output


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.cab0 = nn.Conv2d(num_feat // 2, num_feat // 2, 3, 1, 1)
        self.cab1 = nn.Conv2d(num_feat // 2, num_feat, 3, 1, 1)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # B C 1 1
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y1 = self.attention(x)
        # print(x.shape)
        x_1 = x * y1
        # print(x_1.shape)
        x1, x2 = x_1.chunk(2, dim=1)
        # print(x1.shape)
        x1 = self.cab0(x1)
        # print(x1.shape)
        x_2 = F.gelu(x2) * x1
        x_2 = self.cab1(x_2)
        out = x * x_2  # 改进
        return out



class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = nn.Conv2d(2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out)) * x
        return out


class ASPP(nn.Module):
    def __init__(self, in_channel=512, out_channel=256, squeeze_factor=30, layer=0):
        super(ASPP, self).__init__()
        self.mean = nn.AdaptiveAvgPool2d((1, 1))  # (1,1)means ouput_dim
        self.conv = S_conv(in_channel, out_channel, 3, 1)
        self.layer = layer
    #
        self.atrous_block1 = nn.Conv2d(in_channel, out_channel, 1, 1)
        self.atrous_block3 = nn.Conv2d(in_channel, out_channel, 3, 1, padding=2, dilation=2)
        self.atrous_block6 = nn.Conv2d(in_channel, out_channel, 3, 1, padding=3, dilation=3)
        self.atrous_block12 = nn.Conv2d(in_channel, out_channel, 3, 1, padding=5, dilation=5)
        self.atrous_block18 = nn.Conv2d(in_channel, out_channel, 3, 1, padding=7, dilation=7)

        self.conv_1x1_output = nn.Conv2d(out_channel * 6, out_channel, 1, 1)
        self.bn = nn.BatchNorm2d(in_channel)
        self.conv1x1 = nn.Conv2d(in_channel, out_channel, 1, 1)
        self.act = nn.ReLU(inplace=True)
        self.conv1 = S_conv(in_channel, in_channel)
        self.ca = ChannelAttention(in_channel, squeeze_factor)
        self.sa = SpatialAttentionModule()
        # self.sda =IDynamicSa(in_channel,window_size=7,heads=6)
        #
        self.conv_ = nn.Conv2d(in_channel, in_channel, 3, 1, 1)

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        size = x.shape[2:]
        # if not self.layer:
        #
        #     x = self.sa(x)
        # else:
        #     x = self.sa(x)
        x = self.ca(x)
        x = self.sa(x)
        image_features = self.mean(x)
        image_features = self.conv(self.act(self.conv(image_features)))
        image_features = F.upsample(image_features, size=size, mode='bilinear')

        atrous_block1 = self.atrous_block1(self.act(self.bn(self.atrous_block1(x))))
        atrous_block3 = self.atrous_block3(self.act(self.bn(self.atrous_block3(x))))
        atrous_block6 = self.atrous_block6(self.act(self.bn(self.atrous_block6(x))))
        atrous_block12 = self.atrous_block12(self.act(self.bn(self.atrous_block12(x))))
        atrous_block18 = self.atrous_block18(self.act(self.bn(self.atrous_block18(x))))
        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block3, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        # net = self.ca(net)
        # net = self.sa(net)
        net = self.act(self.bn(self.conv_(net)))

        return to_3d(net)


class Embed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        # _, _, H, W = x.shape
        if self.norm is not None:
            x = self.norm(x)
        return x


class Merge(nn.Module):
    def __init__(self, dim, h, w):
        super(Merge, self).__init__()
        self.conv = nn.Conv2d(dim, dim * 2, kernel_size=2, stride=2, padding=0)
        self.h = h
        self.dim = dim
        self.w = w
        self.norm = nn.BatchNorm2d(dim * 2)

    def forward(self, x):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, self.h, self.w)
        x = self.norm(self.conv(x))

        return x.reshape(B, self.dim * 2, -1).permute(0, 2, 1)


class Expand(nn.Module):
    def __init__(self, dim, h):
        super(Expand, self).__init__()
        self.dim = dim
        self.h = h
        self.conv = nn.ConvTranspose2d(self.dim, self.dim // 2, 2, stride=2)

    def forward(self, x):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, self.h, self.h)
        x = self.conv(x)

        return x.reshape(B, self.dim // 2, -1).permute(0, 2, 1)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.embed = Embed(512)

        self.l1 = nn.Sequential(
            # Block(96, 128, 4, 8, 3, 16),
            # Block(96, 128, 4, 8, 3, 16),
            # Block(96, 128, 4, 8, 3, 16),
            # Block(96, 128, 4, 8, 3, 16)
            Block(96, 128, 4),
            Block(96, 128, 4),
            Block(96, 128, 4),
                                )

        self.l2 = nn.Sequential(
            # Block(192, 64, 4, 8, 3, 16),
            # Block(192, 64, 4, 8, 3, 16),
            # Block(192, 64, 4, 8, 3, 16),
            # Block(192, 64, 4, 8, 3, 16)
            Block(192, 64, 4),
            Block(192, 64, 4),
            Block(192, 64, 4),
                                )

        self.l3 = nn.Sequential(
            # Block(384, 32, 4, 8, 3, 16),
            # Block(384, 32, 4, 8, 3, 16),
            # Block(384, 32, 4, 8, 3, 16),
            # Block(384, 32, 4, 8, 3, 16)
            Block(384, 32, 4),
            Block(384, 32, 4),
            # Block(384, 32, 4),
                                )

        self.l4 = nn.Sequential(
            # Block(768, 16, 2, 8, 3, 16),
            # Block(768, 16, 2, 8, 3, 16),
            # Block(768, 16, 2, 8, 3, 16),
            # Block(768, 16, 2, 8, 3, 16)
            Block(768, 16, 2),
            Block(768, 16, 2),
            Block(768, 16, 2),
            # Block(768, 16, 2),
                                )

        self.m1 = Merge(96, 128, 128)
        self.m2 = Merge(192, 64, 64)
        self.m3 = Merge(384, 32, 32)

        self.p3 = Expand(768, 16)
        self.p2 = Expand(384, 32)
        self.p1 = Expand(192, 64)

        self.d3 = nn.Sequential(
            # Block(384, 32, 4, 8, 3, 16),
            # Block(384, 32, 4, 8, 3, 16),
            # Block(384, 32, 4, 8, 3, 16),
            # Block(384, 32, 4, 8, 3, 16)
            Block(384, 32, 4),
            Block(384, 32, 4),
            # Block(384, 32, 4)
                                )

        self.d2 = nn.Sequential(
            # Block(192, 64, 4, 8, 3, 16),
            # Block(192, 64, 4, 8, 3, 16),
            # Block(192, 64, 4, 8, 3, 16),
            # Block(192, 64, 4, 8, 3, 16)
            Block(192, 64, 4),
            Block(192, 64, 4),
            Block(192, 64, 4),
                                )

        self.d1 = nn.Sequential(
            # Block(96, 128, 4, 8, 3, 16),
            # #Block(96, 128, 4, 8, 3, 16),
            # Block(96, 128, 4, 8, 3, 16),
            # Block(96, 128, 4, 8, 3, 16)
            Block(96, 128, 4),
            Block(96, 128, 4),
            Block(96, 128, 4),

                                )
        #
        self.la_1 = ASPP(96, 96, layer=0)
        self.la_2 = ASPP(192, 192, layer=1)
        self.la_3 = ASPP(384, 384, layer=2)
        self.la_4 = ASPP(768, 768, layer=3)

        #
        self.dbm3 = DeBlock(384)
        self.dbm2 = DeBlock(192)
        self.dbm1 = DeBlock(96)
        #
        self.pas_96 = PSA_p(96, 96)
        self.pas_192 = PSA_p(192, 192)
        self.pas_384 = PSA_p(384, 384)
        # # sobel 模块增
        # self.finalfuss_384 = FinalFuse(384, 192)
        # self.finalfuss_192 = FinalFuse(192, 96)

        self.up = nn.PixelShuffle(4)
        self.seg = nn.Conv2d(6, 1, 1)
        #
        ##
        self.TC_De96 = Trans_CNN_DeBlock(96, 128)
        self.TC_De192 = Trans_CNN_DeBlock(192, 64)
        self.TC_De384 = Trans_CNN_DeBlock(384, 32)
        #
        self.scblk96 = StripConvBlock(96, 96, nn.BatchNorm2d)
        self.scblk192 = StripConvBlock(192, 192, nn.BatchNorm2d)
        self.scblk384 = StripConvBlock(384, 384, nn.BatchNorm2d)

    def forward(self, x):
        B, C, H, W = x.shape

        #
        x_pool  = x
        x = self.embed(x)  # torch.Size([1, 16384, 96])
        #
        x1 = self.l1(x)  # torch.Size([1, 16384, 96])
        xa_1 = self.la_1(x)
        x1_temp = self.TC_De96(x1,xa_1)
        #
        x = self.m1(x1)  # torch.Size([1, 4096, 192])
        xa = self.m1(xa_1)
        #
        x2 = self.l2(x)  # torch.Size([1, 4096, 192])
        xa_2 = self.la_2(xa)
        x2_temp = self.TC_De192(x2, xa_2)
        #
        x = self.m2(x2)  # torch.Size([1, 1024, 384])
        xa = self.m2(xa_2)
        #
        x3 = self.l3(x)  # torch.Size([1, 1024, 384])
        xa_3 = self.la_3(xa)
        x3_temp = self.TC_De384(x3, xa_3)
        #
        x = self.m3(x3)  # torch.Size([1, 256, 768])
        x4 = self.l4(x)  # torch.Size([1, 256, 768])
        #
        x = self.p3(x4)  # torch.Size([1, 1024, 384])
        #
        x = self.d3(x3_temp + x)  # torch.Size([1, 1024, 384])
        #
        x = self.p2(x)  # torch.Size([1, 4096, 192])
        #
        x = self.d2(x + x2_temp)
        #
        x = self.p1(x)  # torch.Size([1, 16384, 96])
        #
        x = self.d1(x + x1_temp)  # 128x128
        x = self.up((x).permute(0, 2, 1).reshape(B, 96, 128, 128))  # torch.Size([1, 6, 512, 512])

        x = self.seg(x)
        return x

from ptflops import get_model_complexity_info
from thop import  profile
from thop import clever_format
if __name__ == '__main__':
    x = torch.rand(1, 3, 512, 512).cuda()


    # y = torch.rand(1, 1024, 384).cuda()
    # dbm = DeBlock(384).cuda()
    part = Net().to("cpu")
    flops, params = profile(part, inputs=(torch.rand(1, 3, 512, 512),))
    flops, params = clever_format([flops, params], '%.3f')
    print('模型参数：', params)
    print('每一个样本浮点运算量：', flops)
    # out = part(x)
    # print(out.shape)
    # out = dbm(y)
    # print(out.shape)
