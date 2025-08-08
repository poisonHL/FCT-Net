import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from Separable_convolution import S_conv
import math
import torch.nn.functional as F
from einops import rearrange, repeat
from itertools import repeat
# from sobel import Gedge_map
import collections.abc
NEG_INF=-1000000
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')
def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse
to_2tuple = _ntuple(2)
class FastLeFF(nn.Module):

    def __init__(self, dim=32, hidden_dim=128, num_heads=8, act_layer=nn.GELU, drop=0.):
        super().__init__()

        self.num_heads = num_heads
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim),
                                     act_layer())
        self.dwconv1 = nn.Sequential(S_conv(hidden_dim, hidden_dim * 2), act_layer())
        # self.dwconv1 = nn.Sequential(DeformableConv2d(hidden_dim, hidden_dim, 3), act_layer(),self.norm())
        self.dwconv2 = nn.Sequential(S_conv(hidden_dim, hidden_dim), act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

    def forward(self, x):
        # bs x hw x c
        bs, hw, c = x.size()
        hh = int(math.sqrt(hw))

        x = self.linear1(x)

        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h=hh, w=hh)
        # bs,hidden_dim,32x32

        x1, x2 = self.dwconv1(x).chunk(2, dim=1)
        x1 = torch.nn.functional.normalize(x1, dim=-1)
        x2 = torch.nn.functional.normalize(x2, dim=-1)
        x1 = self.dwconv2(x2)
        x = F.gelu(x1) * x2

        # flaten
        x = rearrange(x, ' b c h w -> b (h w) c', h=hh, w=hh)

        x = self.linear2(x)

        return x

class DynamicPosBias(nn.Module):
    def __init__(self, dim, num_heads, residual):
        super().__init__()
        self.residual = residual
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim)
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads)
        )
    def forward(self, biases):
        if self.residual:
            pos = self.pos_proj(biases) # 2Wh-1 * 2Ww-1, heads
            pos = pos + self.pos1(pos)
            pos = pos + self.pos2(pos)
            pos = self.pos3(pos)
        else:
            pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos
class Attention(nn.Module):
    r""" Multi-head self attention module with dynamic position bias.

    Args:
        dim (int): Number of input channels.
        group_size (tuple[int]): The height and width of the group.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, resolution,group_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 position_bias=True,flag=0):

        super().__init__()
        self.dim = dim
        self.resolution = resolution
        if flag ==0:
            self.group_size = (7,7)  # Wh, Ww
        else:
            self.group_size = (self.resolution//8, self.resolution//8)  # Wh, Ww
        # self.group_size = group_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.position_bias = position_bias

        if position_bias:
            self.pos = DynamicPosBias(self.dim // 4, self.num_heads, residual=False)

            # generate mother-set
            position_bias_h = torch.arange(1 - self.group_size[0], self.group_size[0])
            position_bias_w = torch.arange(1 - self.group_size[1], self.group_size[1])
            biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w]))  # 2, 2Wh-1, 2W2-1
            biases = biases.flatten(1).transpose(0, 1).float()
            self.register_buffer("biases", biases)

            # get pair-wise relative position index for each token inside the group
            coords_h = torch.arange(self.group_size[0])
            coords_w = torch.arange(self.group_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.group_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.group_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.group_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x,H,W, mask=None):
        """
        Args:
            x: input features with shape of (num_groups*B, N, C)
            mask: (0/-inf) mask with shape of (num_groups, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        self.group_size=(H,W)
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.position_bias:
            pos = self.pos(self.biases)  # 2Wh-1 * 2Ww-1, heads
            # select position bias
            relative_position_bias = pos[self.relative_position_index.view(-1)].view(
                self.group_size[0] * self.group_size[1], self.group_size[0] * self.group_size[1], -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Conv_block, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out // 4, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out // 4)
        self.conv2 = nn.Conv2d(ch_out //4, ch_out, kernel_size=3, padding=1)
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


class Block(nn.Module):
    def __init__(self, num_feat, resolution,mlp_ratio=4,drop_path=0.1,interval = 8, window_size =7):
        super(Block, self).__init__()
        # self.cab = CAB(num_feat, compress_ratio, squeeze_factor)
        self.norm1 = nn.LayerNorm(num_feat)
        self.norm2 = nn.LayerNorm(num_feat)
        self.dim = num_feat
        self.resolution = resolution
        #以下为新增
        self.c1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.norm3 = nn.BatchNorm2d(num_feat)
        self.act = nn.ReLU()
        self.c2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.c3 = nn.Conv2d(num_feat, num_feat, 1, 1, 1)
        self.act2 = nn.ReLU()
        # 增间隔
        self.interval = interval
        self.window_size = window_size
        self.attn1 = Attention(
            num_feat,resolution=self.resolution, group_size=to_2tuple(self.window_size), num_heads=4,
            qkv_bias=True, qk_scale=None, attn_drop=0, proj_drop=0.1,
            position_bias=True,flag=0)
        self.attn2 = Attention(
            num_feat,resolution=self.resolution, group_size=to_2tuple(self.interval), num_heads=4,
            qkv_bias=True, qk_scale=None, attn_drop=0, proj_drop=0.1,
            position_bias=True, flag=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        #以上为新增

        #self.windows_size = window_size #调换位置
        # self.wsa = WindowAttention(num_feat, (window_size, window_size), heads)
        mlp_hidden_dim = int(num_feat * mlp_ratio)
        # self.mlp = DWMlp(in_features=num_feat, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, res=input_resolution)
        self.mlp = FastLeFF(num_feat)
        self.conv_b = Conv_block(num_feat,num_feat)

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(N**0.5)
        #x_1 = self.norm1(x)
        x_res = x
        #以下为新增
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.norm3(self.c2(self.act(self.norm3(self.c1(x))))) # 新增
        x = x.reshape(B, C, -1).permute(0, 2, 1)

        #Cross Attention
        shortcut = x
        x = self.norm1(x)
        x  =x.contiguous().view(B, H, W, C)
        x = x.view(B, H, W, C)

        # padding
        size_par = self.window_size# 7
        pad_l = pad_t = 0
        pad_r = (size_par - W % size_par) % size_par
        pad_b = (size_par - H % size_par) % size_par
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hd, Wd, _ = x.shape

        mask = torch.zeros((1, Hd, Wd, 1), device=x.device)
        if pad_b > 0:
            mask[:, -pad_b:, :, :] = -1
        if pad_r > 0:
            mask[:, :, -pad_r:, :] = -1

        G = Gh= Gw=self.window_size
        x = x.reshape(B, Hd // G, G, Wd // G, G, C).permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.reshape(B * Hd * Wd // G ** 2, G ** 2, C)
        nP = Hd * Wd // G ** 2  # number of partitioning groups
        # attn_mask
        if pad_r > 0 or pad_b > 0:
            mask = mask.reshape(1, Hd // G, G, Wd // G, G, 1).permute(0, 1, 3, 2, 4, 5).contiguous()
            mask = mask.reshape(nP, 1, G * G)
            attn_mask = torch.zeros((nP, G * G, G * G), device=x.device)
            attn_mask = attn_mask.masked_fill(mask < 0, NEG_INF)
        else:
            attn_mask = None

        # multi-head self-attention
        x = self.attn1(x, Gh, Gw, mask=attn_mask)  # nW*B, G*G,-
        x = x.reshape(B, Hd // G, Wd // G, G, G, C).permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.reshape(B, Hd, Wd, C)
        # remove padding
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)
        x = x.contiguous().view(B, H * W, C)
        #MLP
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm1(x)))

        # flag =1
        shortcut1 = x
        x = self.norm1(x)
        x = x.contiguous().view(B, H, W, C)
        # padding
        size_par = self.interval  # 8
        pad_l = pad_t = 0
        pad_r = (size_par - W % size_par) % size_par  # 0
        pad_b = (size_par - H % size_par) % size_par  # 0
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hd, Wd, _ = x.shape

        mask = torch.zeros((1, Hd, Wd, 1), device=x.device)
        if pad_b > 0:
            mask[:, -pad_b:, :, :] = -1
        if pad_r > 0:
            mask[:, :, -pad_r:, :] = -1  # 用不到
        I, Gh, Gw = self.interval, Hd // self.interval, Wd // self.interval
        x = x.reshape(B, Gh, I,  Gw,I , C).permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(B * I * I , Gh *Gw, C)
        nP= I**2
        # attn_mask
        if pad_r > 0 or pad_b > 0:
            mask = mask.reshape(1, Gh, I, Gw, I, 1).permute(0, 2, 4, 1, 3, 5).contiguous()
            mask = mask.reshape(nP, 1, Gh * Gw)
            attn_mask = torch.zeros((nP, Gh * Gw, Gh * Gw), device=x.device)
            attn_mask = attn_mask.masked_fill(mask < 0, NEG_INF)
        else:
            attn_mask = None
        # multi-head self-attention
        x = self.attn2(x,Gh, Gw, mask=None)  # nW*B, G*G,
        x = x.reshape(B, I, I, Gh, Gw, C).permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.reshape(B, Hd, Wd, C)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C).contiguous()


        # MLP
        x = shortcut1 + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm1(x)))
        #增3*3卷积块
        x = to_3d(self.conv_b(to_4d(x,H,W)))
        x = x_res + x
        return x


if __name__ == '__main__':
    x = torch.rand(1, 16384, 96).cuda()
    # y = torch.rand(1, 128, 128, 96).cuda()
    # model = CAB(96, 3, 16).cuda()
    model = Block(96, 128, 4,).cuda()
    # model = WindowAttention(96, (4, 4), 8).cuda()
    # out = model(x)
    # out = window_partition(y, 4).reshape(1024, 16, 96)
    out = model(x)
    print(out.shape)
