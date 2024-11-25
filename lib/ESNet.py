import timm
import torch
import torch.nn as nn
from ptflops import get_model_complexity_info
from thop import profile, clever_format
import torch.nn.functional as F

from lib.Modules import SkeletonEncoder, DimensionalReduction, SkeletonGuidanceFusion, PORM
from lib.SMT import smt_t


class ESNet(nn.Module):
    def __init__(self, ):
        super(ESNet, self).__init__()

        print('--> using efficientnet-b3 right now')

        self.context_encoder = timm.create_model(model_name='tf_efficientnet_b3.ns_jft_in1k', features_only=True,
                                                 pretrained=False, out_indices=range(0, 5))
        pretrained_dict = torch.load('/media/omnisky/data/rp/COD_TILNet/pre/tf_efficientnet_b3_ns-9d44bf68.pth')
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.context_encoder.state_dict()}
        self.context_encoder.load_state_dict(pretrained_dict)
        # self.context_encoder = smt_t(pretrained=True)
        in_channel_list = [48, 136, 384]

        self.skeleton_encoder = SkeletonEncoder()

        self.dr3 = DimensionalReduction(in_channel=in_channel_list[0], out_channel=32)
        self.dr4 = DimensionalReduction(in_channel=in_channel_list[1], out_channel=32)
        self.dr5 = DimensionalReduction(in_channel=in_channel_list[2], out_channel=32)

        self.sgf = SkeletonGuidanceFusion(channel=32)

        self.porm = PORM(channels=32)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.predtrans2 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.predtrans3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.predtrans4 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.predtrans5 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        size = x.size()[2:]

        endpoints = self.context_encoder(x)

        x3 = endpoints[2]
        x4 = endpoints[3]
        x5 = endpoints[4]

        xr3 = self.dr3(x3)
        xr4 = self.dr4(x4)
        xr5 = self.dr5(x5)

        xg, pg = self.skeleton_encoder(x)

        # decoder
        zt3, zt4, zt5 = self.sgf(xr3, xr4, xr5, xg)
        s2, s3, s4, pose = self.porm(zt3, zt4, zt5)
        pg = self.upsample(pg)

        s2 = F.interpolate(self.predtrans2(s2), size=size, mode='bilinear', align_corners=True)
        s3 = F.interpolate(self.predtrans3(s3), size=size, mode='bilinear', align_corners=True)
        s4 = F.interpolate(self.predtrans4(s4), size=size, mode='bilinear', align_corners=True)
        pose = F.interpolate(self.predtrans5(pose), size=size, mode='bilinear', align_corners=True)

        return s2, s3, s4, pose, pg


if __name__ == '__main__':
    net = ESNet().eval()
    inputs = torch.randn(1, 3, 352, 352)
    outs = net(inputs)
    print(outs[0].shape)

    params, flops = profile(net, inputs=(inputs,))
    flops, params = clever_format([flops, params], "%.2f")
    print(flops, params)

