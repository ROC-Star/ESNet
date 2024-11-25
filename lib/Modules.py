"""
@FileName:Modules.py
@Author:ROC
@Time:2023/8/22 14:22
"""
import math
import numbers

import torch
from einops import rearrange
from timm.models.layers import trunc_normal_
from torch.nn import functional as F
import torch.nn as nn

eps = 1e-12


class _DSConv(nn.Module):
    """Depthwise Separable Convolutions"""

    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(_DSConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, dw_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(dw_channels),
            nn.ReLU(True),
            nn.Conv2d(dw_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


def _make_layer(block, inplanes, planes, blocks, t=6, stride=1):
    layers = []
    layers.append(block(inplanes, planes, stride, t))
    for i in range(1, blocks):
        layers.append(block(planes, planes, 1, t))
    return nn.Sequential(*layers)


class _DWConv(nn.Module):
    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(_DWConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, out_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class LinearBottleneck(nn.Module):
    """LinearBottleneck used in MobileNetV2"""

    def __init__(self, in_channels, out_channels, t=6, stride=2, **kwargs):
        super(LinearBottleneck, self).__init__()
        self.use_shortcut = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(
            # pw
            _DSConv(in_channels, in_channels * t, 1),
            # dw
            _DWConv(in_channels * t, in_channels * t, stride),
            # pw-linear
            nn.Conv2d(in_channels * t, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            # TripletAttention(out_channels)
        )

    def forward(self, x):
        out = self.block(x)
        if self.use_shortcut:
            out = x + out
        return out


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class BasicBlock(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, keep_3x3=False):
        super(BasicBlock, self).__init__()
        assert stride in [1, 2]

        hidden_dim = inp // expand_ratio
        if hidden_dim < oup / 6.:
            hidden_dim = math.ceil(oup / 6.)
            hidden_dim = _make_divisible(hidden_dim, 16)  # + 16

        # self.relu = nn.ReLU6(inplace=True)
        self.identity = False
        self.identity_div = 1
        self.expand_ratio = expand_ratio
        if expand_ratio == 2:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, 1, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(oup, oup, 3, stride, 1, groups=oup, bias=False),
                nn.BatchNorm2d(oup),
            )
        elif inp != oup and stride == 1 and keep_3x3 == False:
            self.conv = nn.Sequential(
                # pw-linear
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True),
            )
        elif inp != oup and stride == 2 and keep_3x3 == False:
            self.conv = nn.Sequential(
                # pw-linear
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(oup, oup, 3, stride, 1, groups=oup, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            if keep_3x3 == False:
                self.identity = True
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, 1, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU6(inplace=True),
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # nn.ReLU6(inplace=True),
                # pw
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(oup, oup, 3, 1, 1, groups=oup, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        out = self.conv(x)

        if self.identity:
            shape = x.shape
            id_tensor = x[:, :shape[1] // self.identity_div, :, :]
            # id_tensor = torch.cat([x[:,:shape[1]//self.identity_div,:,:],torch.zeros(shape)[:,shape[1]//self.identity_div:,:,:].cuda()],dim=1)
            # import pdb; pdb.set_trace()
            out[:, :shape[1] // self.identity_div, :, :] = out[:, :shape[1] // self.identity_div, :, :] + id_tensor
            return out  # + x
        else:
            return out


class ConvBR(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1):
        super(ConvBR, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.GELU()
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class DimensionalReduction(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DimensionalReduction, self).__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, groups=in_channel, padding=1),
            nn.BatchNorm2d(in_channel),
            nn.GELU(),
            ConvBR(in_channel, out_channel, 1, padding=0)
            # ConvBR(out_channel, out_channel, 3, padding=1)
        )

    def forward(self, x):
        return self.reduce(x)


class SkeletonEncoder(nn.Module):
    def __init__(self):
        super(SkeletonEncoder, self).__init__()
        self.conv1 = _make_layer(BasicBlock, 3, 64, blocks=1, t=3, stride=2)
        self.conv2 = _make_layer(BasicBlock, 64, 64, blocks=2, t=2, stride=2)
        # self.sc1 = ScConv(64)
        self.conv3 = _make_layer(BasicBlock, 64, 32, blocks=2, t=2, stride=2)
        # self.sc2 = ScConv(32)

        self.conv_out = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)
        self.conv192_64 = ConvBR(96, 64, 1, 1, 0)
        self.conv128_64 = ConvBR(128, 32, 1, 1, 0)

    def forward(self, x):
        feat = self.conv1(x)

        feat1 = self.conv2(feat)
        # feat1 = self.sc1(feat1)

        xg = self.conv3(feat1)
        # xg = self.sc2(xg)

        xg_up = F.interpolate(xg, size=feat1.size()[2:], mode='bilinear', align_corners=True)
        xg_f = torch.cat([xg_up, feat1], dim=1)
        xg_f = self.conv192_64(xg_f)
        xg_f_up = F.interpolate(xg_f, size=feat.size()[2:], mode='bilinear', align_corners=True)
        pg = torch.cat([xg_f_up, feat], dim=1)
        pg_1 = self.conv128_64(pg)

        pg_out = self.conv_out(pg_1)
        return pg_1, pg_out


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv_0 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.qkv1conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.qkv2conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.qkv3conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, mask=None):
        b, c, h, w = x.shape
        q = self.qkv1conv(self.qkv_0(x))
        k = self.qkv2conv(self.qkv_1(x))
        v = self.qkv3conv(self.qkv_2(x))
        if mask is not None:
            q = q * mask
            k = k * mask

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class DWFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias=False):
        super(DWFFN, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class Layernorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(Layernorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class SGA(nn.Module):
    def __init__(self, dim=128, num_heads=8, ffn_expansion_factor=4, bias=False, LayerNorm_type='WithBias'):
        super(SGA, self).__init__()
        self.norm1 = Layernorm(dim, LayerNorm_type)
        self.attn = SelfAttention(dim, num_heads, bias)
        self.norm2 = Layernorm(dim, LayerNorm_type)
        self.ffn = DWFFN(dim, ffn_expansion_factor, bias)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x


class SkeletonGuidanceFusion(nn.Module):
    def __init__(self, channel):
        super(SkeletonGuidanceFusion, self).__init__()
        self.downsample4 = nn.Upsample(scale_factor=1 / 4, mode='bilinear', align_corners=True)
        self.downsample8 = nn.Upsample(scale_factor=1 / 8, mode='bilinear', align_corners=True)
        self.downsample16 = nn.Upsample(scale_factor=1 / 16, mode='bilinear', align_corners=True)

        self.sgs3 = ScConv(channel)
        self.sgs4 = ScConv(channel)
        self.sgs5 = ScConv(channel)
        self.attn = SGA(dim=channel, num_heads=8)
        self.attn1 = SGA(dim=channel, num_heads=8)
        self.attn2 = SGA(dim=channel, num_heads=8)

    def forward(self, xr3, xr4, xr5, xg):

        q3 = self.attn(xr3, self.downsample4(xg))
        q4 = self.attn1(xr4, self.downsample8(xg))
        q5 = self.attn2(xr5, self.downsample16(xg))
        zt3 = self.sgs3(q3)
        zt4 = self.sgs4(q4)
        zt5 = self.sgs5(q5)

        return zt3, zt4, zt5

    def gradient_induced_feature_grouping(self, xr, xg, M):
        if not M in [1, 2, 4, 8, 16, 32]:
            raise ValueError("Invalid Group Number!: must be one of [1, 2, 4, 8, 16, 32]")

        if M == 1:
            return torch.cat((xr, xg), 1)

        xr_g = torch.chunk(xr, M, dim=1)
        xg_g = torch.chunk(xg, M, dim=1)
        foo = list()
        for i in range(M):
            foo.extend([xr_g[i], xg_g[i]])

        return torch.cat(foo, 1)


class EmRouting2d(nn.Module):
    def __init__(self, A, B, caps_size, kernel_size=3, stride=1, padding=1, iters=3, final_lambda=1e-2):
        super(EmRouting2d, self).__init__()
        self.A = A
        self.B = B
        self.psize = caps_size
        self.mat_dim = int(caps_size ** 0.5)

        self.k = kernel_size
        self.kk = kernel_size ** 2
        self.kkA = self.kk * A

        self.stride = stride
        self.pad = padding

        self.iters = iters

        self.W = nn.Parameter(torch.FloatTensor(self.kkA, B, self.mat_dim, self.mat_dim))
        nn.init.kaiming_uniform_(self.W.data)

        self.beta_u = nn.Parameter(torch.FloatTensor(1, 1, B, 1))
        self.beta_a = nn.Parameter(torch.FloatTensor(1, 1, B))
        nn.init.constant_(self.beta_u, 0)
        nn.init.constant_(self.beta_a, 0)

        self.final_lambda = final_lambda
        self.ln_2pi = math.log(2 * math.pi)

    def m_step(self, v, a_in, r):
        # v: [b, l, kkA, B, psize]
        # a_in: [b, l, kkA]
        # r: [b, l, kkA, B, 1]
        b, l, _, _, _ = v.shape

        # r: [b, l, kkA, B, 1]
        r = r * a_in.view(b, l, -1, 1, 1)
        # r_sum: [b, l, 1, B, 1]
        r_sum = r.sum(dim=2, keepdim=True)
        # coeff: [b, l, kkA, B, 1]
        coeff = r / (r_sum + eps)

        # mu: [b, l, 1, B, psize]
        mu = torch.sum(coeff * v, dim=2, keepdim=True)
        # sigma_sq: [b, l, 1, B, psize]
        sigma_sq = torch.sum(coeff * (v - mu) ** 2, dim=2, keepdim=True) + eps

        # [b, l, B, 1]
        r_sum = r_sum.squeeze(2)
        # [b, l, B, psize]
        sigma_sq = sigma_sq.squeeze(2)
        # [1, 1, B, 1] + [b, l, B, psize] * [b, l, B, 1]
        cost_h = (self.beta_u + torch.log(sigma_sq.sqrt())) * r_sum
        # cost_h = (torch.log(sigma_sq.sqrt())) * r_sum

        # [b, l, B]
        a_out = torch.sigmoid(self.lambda_ * (self.beta_a - cost_h.sum(dim=3)))
        # a_out = torch.sigmoid(self.lambda_*(-cost_h.sum(dim=3)))

        return a_out, mu, sigma_sq

    def e_step(self, v, a_out, mu, sigma_sq):
        b, l, _ = a_out.shape
        # v: [b, l, kkA, B, psize]
        # a_out: [b, l, B]
        # mu: [b, l, 1, B, psize]
        # sigma_sq: [b, l, B, psize]

        # [b, l, 1, B, psize]
        sigma_sq = sigma_sq.unsqueeze(2)

        ln_p_j = -0.5 * torch.sum(torch.log(sigma_sq * self.ln_2pi), dim=-1) \
                 - torch.sum((v - mu) ** 2 / (2 * sigma_sq), dim=-1)

        # [b, l, kkA, B]
        ln_ap = ln_p_j + torch.log(a_out.view(b, l, 1, self.B))
        # [b, l, kkA, B]
        r = torch.softmax(ln_ap, dim=-1)
        # [b, l, kkA, B, 1]
        return r.unsqueeze(-1)

    def forward(self, a_in, pose):
        # pose: [batch_size, A, psize]
        # a: [batch_size, A]
        batch_size = a_in.shape[0]

        # a: [b, A, h, w]
        # pose: [b, A*psize, h, w]
        b, _, h, w = a_in.shape

        # [b, A*psize*kk, l]
        pose = F.unfold(pose, self.k, stride=self.stride, padding=self.pad)  # 手动实现卷积滑窗
        l = pose.shape[-1]
        # [b, A, psize, kk, l]
        pose = pose.view(b, self.A, self.psize, self.kk, l)
        # [b, l, kk, A, psize]
        pose = pose.permute(0, 4, 3, 1, 2).contiguous()
        # [b, l, kkA, psize]
        pose = pose.view(b, l, self.kkA, self.psize)
        # [b, l, kkA, 1, mat_dim, mat_dim]
        pose = pose.view(batch_size, l, self.kkA, self.mat_dim, self.mat_dim).unsqueeze(3)

        # [b, l, kkA, B, mat_dim, mat_dim]
        pose_out = torch.matmul(pose, self.W)

        # [b, l, kkA, B, psize]
        v = pose_out.view(batch_size, l, self.kkA, self.B, -1)

        # [b, kkA, l]
        a_in = F.unfold(a_in, self.k, stride=self.stride, padding=self.pad)
        # [b, A, kk, l]
        a_in = a_in.view(b, self.A, self.kk, l)
        # [b, l, kk, A]
        a_in = a_in.permute(0, 3, 2, 1).contiguous()
        # [b, l, kkA]
        a_in = a_in.view(b, l, self.kkA)

        r = a_in.new_ones(batch_size, l, self.kkA, self.B, 1)
        for i in range(self.iters):
            # this is from open review
            self.lambda_ = self.final_lambda * (1 - 0.95 ** (i + 1))
            a_out, pose_out, sigma_sq = self.m_step(v, a_in, r)
            if i < self.iters - 1:
                r = self.e_step(v, a_out, pose_out, sigma_sq)

        # [b, l, B*psize]
        pose_out = pose_out.squeeze(2).view(b, l, -1)
        # [b, B*psize, l]
        pose_out = pose_out.transpose(1, 2)
        # [b, B, l]
        a_out = a_out.transpose(1, 2).contiguous()

        oh = ow = math.floor(l ** (1 / 2))

        a_out = a_out.view(b, -1, oh, ow)
        pose_out = pose_out.view(b, -1, oh, ow)

        return a_out, pose_out


class PORM(nn.Module):
    def __init__(self, channels):
        super(PORM, self).__init__()
        self.conv_trans = nn.Conv2d(channels * 3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_trans = nn.BatchNorm2d(64)

        self.num_caps = 8
        planes = 16
        last_size = 6

        self.conv_m = nn.Conv2d(64, self.num_caps, kernel_size=5, stride=1, padding=1, bias=False)
        self.conv_pose = nn.Conv2d(64, self.num_caps * 16, kernel_size=5, stride=1, padding=1, bias=False)

        self.bn_m = nn.BatchNorm2d(self.num_caps)
        self.bn_pose = nn.BatchNorm2d(self.num_caps * 16)

        self.emrouting = EmRouting2d(self.num_caps, self.num_caps, 16, kernel_size=3, stride=2, padding=1)
        self.bn_caps = nn.BatchNorm2d(self.num_caps * planes)

        self.conv_rec = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_rec = nn.BatchNorm2d(64)

        self.conv1 = nn.Conv2d(96, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(96, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(96, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        self.fuse1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                                   nn.Conv2d(64, 64, kernel_size=3, groups=64, padding=1),
                                   nn.BatchNorm2d(64), nn.GELU())
        self.fuse2 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1),
                                   nn.Conv2d(64, 64, kernel_size=3, groups=64, padding=1),
                                   nn.BatchNorm2d(64), nn.GELU())
        self.fuse3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1),
                                   nn.Conv2d(64, 64, kernel_size=3, groups=64, padding=1),
                                   nn.BatchNorm2d(64), nn.GELU())

    def forward(self, input1, input2, input3):

        if input3.size()[2:] != input1.size()[2:]:
            input3 = F.interpolate(input3, size=input1.size()[2:], mode='bilinear')
        if input2.size()[2:] != input1.size()[2:]:
            input2 = F.interpolate(input2, size=input1.size()[2:], mode='bilinear')

        input_fuse = torch.cat((input1, input2, input3), 1)

        # conv
        input_fuse = F.relu(self.bn_trans(self.conv_trans(input_fuse)))

        # primary caps
        m, pose = self.conv_m(input_fuse), self.conv_pose(input_fuse)

        m, pose = torch.sigmoid(self.bn_m(m)), self.bn_pose(pose)

        # caps
        m, pose = self.emrouting(m, pose)
        pose = self.bn_caps(pose)

        # reconstruction
        pose = F.relu(self.bn_rec(self.conv_rec(pose)))

        if pose.size()[2:] != input1.size()[2:]:
            pose1 = F.interpolate(pose, size=input1.size()[2:], mode='bilinear')
        if pose.size()[2:] != input2.size()[2:]:
            pose2 = F.interpolate(pose, size=input2.size()[2:], mode='bilinear')
        if pose.size()[2:] != input3.size()[2:]:
            pose3 = F.interpolate(pose, size=input3.size()[2:], mode='bilinear')

        out1 = torch.cat((input1, pose1), 1)
        out2 = torch.cat((input2, pose2), 1)
        out3 = torch.cat((input3, pose3), 1)

        out1 = F.relu(self.bn1(self.conv1(out1)))
        out2 = F.relu(self.bn2(self.conv2(out2)))
        out3 = F.relu(self.bn3(self.conv3(out3)))

        # out1 = F.interpolate(out1, size=out2.size()[2:], mode='bilinear')
        # out2 = self.fuse2(out2 * out1) + out2
        # out2 = F.interpolate(out2, size=out3.size()[2:], mode='bilinear')
        # out3 = self.fuse3(out3 * out2) + out3

        out1 = F.interpolate(out1, size=out2.size()[2:], mode='bilinear')
        out2 = self.fuse2(torch.cat([out2, out1], dim=1)) + out2
        out2 = F.interpolate(out2, size=out3.size()[2:], mode='bilinear')
        out3 = self.fuse3(torch.cat([out3, out2], dim=1)) + out3

        return out1, out2, out3, pose


# class SCBottleneck(nn.Module):
#
#     expansion = 1
#     pooling_r = 4  # down-sampling rate of the avg pooling layer in the K3 path of SC-Conv.
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None,
#                  cardinality=1, bottleneck_width=32,
#                  avd=False, dilation=1, is_first=False,
#                  norm_layer=None):
#         super(SCBottleneck, self).__init__()
#         group_width = int(planes * (bottleneck_width / 64.)) * cardinality
#         self.conv1_a = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
#         self.bn1_a = norm_layer(group_width)
#         self.conv1_b = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
#         self.bn1_b = norm_layer(group_width)
#         self.avd = avd and (stride > 1 or is_first)
#
#         if self.avd:
#             self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
#             stride = 1
#
#         self.k1 = nn.Sequential(
#             nn.Conv2d(
#                 group_width, group_width, kernel_size=3, stride=stride,
#                 padding=dilation, dilation=dilation,
#                 groups=cardinality, bias=False),
#             norm_layer(group_width),
#         )
#
#         self.scconv = SCConv(
#             group_width, group_width, stride=stride,
#             padding=dilation, dilation=dilation,
#             groups=cardinality, pooling_r=self.pooling_r, norm_layer=norm_layer)
#
#         self.conv3 = nn.Conv2d(
#             group_width * 2, planes, kernel_size=1, bias=False)
#         self.bn3 = norm_layer(planes)
#
#         self.relu = nn.ReLU(inplace=True)
#         self.conv = nn.Sequential(
#             nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(planes),
#             nn.ReLU(inplace=True)
#         )
#         self.downsample = downsample
#         self.dilation = dilation
#         self.stride = stride
#
#     def forward(self, x):
#         residual = x
#
#         out_a = self.conv1_a(x)
#         out_a = self.bn1_a(out_a)
#         out_b = self.conv1_b(x)
#         out_b = self.bn1_b(out_b)
#         out_a = self.relu(out_a)
#         out_b = self.relu(out_b)
#
#         out_a = self.k1(out_a)
#         out_b = self.scconv(out_b)
#         out_a = self.relu(out_a)
#         out_b = self.relu(out_b)
#
#         if self.avd:
#             out_a = self.avd_layer(out_a)
#             out_b = self.avd_layer(out_b)
#
#         out = self.conv3(torch.cat([out_a, out_b], dim=1))
#         out = self.bn3(out)
#
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         out += residual
#         out = self.relu(out)
#         out = self.conv(out)
#
#         return out


class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num: int,
                 group_num: int = 16,
                 eps: float = 1e-10
                 ):
        super(GroupBatchnorm2d, self).__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.weight = nn.Parameter(torch.randn(c_num, 1, 1))
        self.bias = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias


class SRU(nn.Module):
    def __init__(self,
                 oup_channels: int,
                 group_num: int = 16,
                 gate_treshold: float = 0.5,
                 torch_gn: bool = False
                 ):
        super().__init__()

        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num) if torch_gn else GroupBatchnorm2d(
            c_num=oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold
        self.sigomid = nn.Sigmoid()

    def forward(self, x):
        gn_x = self.gn(x)
        w_gamma = self.gn.weight / torch.sum(self.gn.weight)
        w_gamma = w_gamma.view(1, -1, 1, 1)
        reweigts = self.sigomid(gn_x * w_gamma)
        # Gate
        info_mask = reweigts >= self.gate_treshold
        noninfo_mask = reweigts < self.gate_treshold
        x_1 = info_mask * gn_x
        x_2 = noninfo_mask * gn_x
        x = self.reconstruct(x_1, x_2)
        return x

    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)


class CRU(nn.Module):
    '''
    alpha: 0<alpha<1
    '''

    def __init__(self,
                 op_channel: int,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.up_channel = up_channel = int(alpha * op_channel)
        self.low_channel = low_channel = op_channel - up_channel
        self.squeeze1 = nn.Conv2d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)
        # up
        self.GWC = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=group_kernel_size, stride=1,
                             padding=group_kernel_size // 2, groups=group_size)
        self.PWC1 = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=1, bias=False)
        # low
        self.PWC2 = nn.Conv2d(low_channel // squeeze_radio, op_channel - low_channel // squeeze_radio, kernel_size=1,
                              bias=False)
        self.advavg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # Split
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)
        # Transform
        Y1 = self.GWC(up) + self.PWC1(up)
        Y2 = torch.cat([self.PWC2(low), low], dim=1)
        # Fuse
        out = torch.cat([Y1, Y2], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        return out1 + out2


class ScConv(nn.Module):
    def __init__(self,
                 op_channel: int,
                 group_num: int = 4,
                 gate_treshold: float = 0.5,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.SRU = SRU(op_channel,
                       group_num=group_num,
                       gate_treshold=gate_treshold)
        self.CRU = CRU(op_channel,
                       alpha=alpha,
                       squeeze_radio=squeeze_radio,
                       group_size=group_size,
                       group_kernel_size=group_kernel_size)

    def forward(self, x):
        x = self.SRU(x)
        x = self.CRU(x)
        return x
