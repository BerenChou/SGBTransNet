import torch
from models.utils import SATransformer, CATransformer
import torch.nn as nn
from mmcv.cnn.bricks.conv_module import ConvModule
from einops import rearrange


class SFRTrans(nn.Module):
    """ backbone_dim: The number of channels of feature maps output from the backbone (UNet Encoder).
        seq_dim: The dimension after the sequentialization.
        feature_map_size: The spatial size (int) of the feature map input to SFRTrans.
        patch_size: The patch size for the sequentialization.
        ffn_dim, head_dim, head_num, dropout, depth: See in utils.py """
    def __init__(self, backbone_dim, seq_dim, feature_map_size, patch_size,
                 ffn_dim, head_dim, head_num=4, dropout=0., depth=3):
        super(SFRTrans, self).__init__()
        self.feature_map_size = feature_map_size
        self.patch_size = patch_size

        self.seq_conv_E = ConvModule(backbone_dim, seq_dim, kernel_size=patch_size, stride=patch_size,
                                     norm_cfg=dict(type='BN'))
        self.seq_conv_D = ConvModule(backbone_dim, seq_dim, kernel_size=patch_size, stride=patch_size,
                                     norm_cfg=dict(type='BN'))
        self.shared_pe = nn.Parameter(torch.zeros(1, (feature_map_size // patch_size) ** 2, seq_dim))

        self.catransformer = CATransformer(seq_dim, ffn_dim=ffn_dim, head_dim=head_dim,
                                           head_num=head_num, dropout=dropout)
        self.satransformer = SATransformer(head_dim * head_num, ffn_dim=ffn_dim, head_dim=head_dim,
                                           head_num=head_num, dropout=dropout, depth=depth)

        self.conv1_1 = ConvModule(head_dim * head_num, backbone_dim, kernel_size=1, norm_cfg=dict(type='BN'))

        if patch_size > 1:
            self.output_upsample = nn.Upsample(scale_factor=patch_size)

    def forward(self, E, D):
        seq_E = rearrange(self.seq_conv_E(E), 'b c h w -> b (h w) c') + self.shared_pe
        seq_D = rearrange(self.seq_conv_D(D), 'b c h w -> b (h w) c') + self.shared_pe

        Z_nplus1 = self.satransformer(self.catransformer(seq_E, seq_D))

        output = self.conv1_1(rearrange(Z_nplus1, 'b (h w) c -> b c h w',
                                        h=self.feature_map_size // self.patch_size))

        if self.patch_size > 1:
            output = self.output_upsample(output)

        return output


if __name__ == '__main__':
    sfrtrans = SFRTrans(backbone_dim=512, seq_dim=256, feature_map_size=28, patch_size=1,
                        ffn_dim=256, head_dim=64).cuda()

    _E = torch.rand(4, 512, 28, 28).cuda()
    _D = torch.rand(4, 512, 28, 28).cuda()

    sfrtrans_output = sfrtrans(_E, _D)

    print(sfrtrans_output.shape)
    print('debugger')
