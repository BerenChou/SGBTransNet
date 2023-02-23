import torch.nn as nn
import torch
from mmseg.models.builder import BACKBONES
from mmcv.cnn.bricks.conv_module import ConvModule
from models.scem import SCEM
from mmcv.runner import ModuleList
import numpy as np
from mmseg.models.backbones.vit import TransformerEncoderLayer


# ConvModule * 2
class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            ConvModule(in_ch, out_ch, kernel_size=3, padding=1, norm_cfg=dict(type='BN')),
            ConvModule(out_ch, out_ch, kernel_size=3, padding=1, norm_cfg=dict(type='BN')),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


# Upsample_2 + ConvModule
class up_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            ConvModule(in_ch, out_ch, kernel_size=3, padding=1, norm_cfg=dict(type='BN')),
        )

    def forward(self, x):
        x = self.up(x)
        return x


class UNetEncoderToTransformer(nn.Module):
    def __init__(self, in_channels=1024, hidden_size=768, n_patches=484, num_layers=4, num_heads=4,
                 mlp_ratio=4, attn_drop_rate=0., drop_rate=0.1, drop_path_rate=0., norm_cfg=None):
        super(UNetEncoderToTransformer, self).__init__()
        if norm_cfg is None:
            norm_cfg = dict(type='LN', eps=1e-6, requires_grad=True)

        self.n_patches = n_patches

        self.patch_embedding = nn.Conv2d(in_channels=in_channels, out_channels=hidden_size, kernel_size=(1, 1))
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, hidden_size))

        self.transformer_layers = ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        for i in range(num_layers):
            self.transformer_layers.append(
                TransformerEncoderLayer(embed_dims=hidden_size, num_heads=num_heads,
                                        feedforward_channels=mlp_ratio * hidden_size,
                                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                        drop_path_rate=dpr[i], norm_cfg=norm_cfg)
            )

        self.conv_hiddenSize2inchannels = ConvModule(in_channels=hidden_size, out_channels=in_channels,
                                                     kernel_size=(1, 1), norm_cfg=dict(type='BN'))

    def forward(self, x):
        x = self.patch_embedding(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = x + self.position_embeddings

        for tr_layer in self.transformer_layers:
            x = tr_layer(x)

        x = x.permute(0, 2, 1)
        h, w = int(np.sqrt(self.n_patches)), int(np.sqrt(self.n_patches))
        x = x.contiguous().view(x.shape[0], x.shape[1], h, w)
        x = self.conv_hiddenSize2inchannels(x)
        return x


@BACKBONES.register_module()
class SGBTransNet(nn.Module):
    def __init__(self, img_size, dpcca_ratio=2, hidden_size_SATr=768):
        super(SGBTransNet, self).__init__()
        base_channel = 64
        channels_list = [base_channel, base_channel * 2, base_channel * 4, base_channel * 8, base_channel * 16]
        # [64, 128, 256, 512, 1024]
        #   0    1    2    3     4

        feature_maps_size_list = [0, 1, img_size // 2, img_size // 4, img_size // 8]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(3, channels_list[0])  # 3, 64
        self.Conv2 = conv_block(channels_list[0], channels_list[1])  # (64, 128)
        self.Conv3 = conv_block(channels_list[1], channels_list[2])  # (128, 256)
        self.Conv4 = conv_block(channels_list[2], channels_list[3])  # (256, 512)
        self.Conv5 = conv_block(channels_list[3], channels_list[4])  # (512, 1024)

        self.Up5 = up_conv(channels_list[4], channels_list[3])  # (1024, 512)
        self.scem4 = SCEM(dpcca_ratio, backbone_dim=channels_list[3], seq_dim=channels_list[2],
                          feature_map_size=feature_maps_size_list[4], patch_size=1, ffn_dim=channels_list[2],
                          head_dim=channels_list[2] // 4, head_num=4, dropout=0., depth=3)
        self.Up_conv5 = conv_block(channels_list[4], channels_list[3])  # (1024, 512)

        self.Up4 = up_conv(channels_list[3], channels_list[2])  # (512, 256)
        self.scem3 = SCEM(dpcca_ratio, backbone_dim=channels_list[2], seq_dim=channels_list[1],
                          feature_map_size=feature_maps_size_list[3], patch_size=2, ffn_dim=channels_list[1],
                          head_dim=channels_list[1] // 4, head_num=4, dropout=0., depth=3)
        self.Up_conv4 = conv_block(channels_list[3], channels_list[2])  # (512, 256)

        self.Up3 = up_conv(channels_list[2], channels_list[1])  # (256, 128)
        self.scem2 = SCEM(dpcca_ratio, backbone_dim=channels_list[1], seq_dim=channels_list[0],
                          feature_map_size=feature_maps_size_list[2], patch_size=4, ffn_dim=channels_list[0],
                          head_dim=channels_list[0] // 4, head_num=4, dropout=0., depth=3)
        self.Up_conv3 = conv_block(channels_list[2], channels_list[1])  # (256, 128)

        self.Up2 = up_conv(channels_list[1], channels_list[0])  # (128, 64)

        self.Up_conv2 = nn.Sequential(
            ConvModule(channels_list[1], channels_list[1], kernel_size=3, padding=1, norm_cfg=dict(type='BN')),
            ConvModule(channels_list[1], channels_list[0], kernel_size=3, padding=1, norm_cfg=dict(type='BN')),
            ConvModule(channels_list[0], channels_list[0], kernel_size=3, padding=1, norm_cfg=dict(type='BN')),
            ConvModule(channels_list[0], channels_list[0], kernel_size=3, padding=1, norm_cfg=dict(type='BN'))
        )

        self.SATransformer = UNetEncoderToTransformer(in_channels=channels_list[4], hidden_size=hidden_size_SATr,
                                                      n_patches=(img_size // 16) ** 2)

        print('******** SGBTransNet has been constructed. ********')

    def forward(self, x):
        # Encoder below
        e1 = self.Conv1(x)
        e2 = self.Maxpool1(e1)

        e2 = self.Conv2(e2)
        e3 = self.Maxpool2(e2)

        e3 = self.Conv3(e3)
        e4 = self.Maxpool3(e3)

        e4 = self.Conv4(e4)
        e5 = self.Maxpool4(e4)

        e5 = self.Conv5(e5)

        e5 = self.SATransformer(e5)

        # Decoder below
        d5 = self.Up5(e5)
        d5 = self.scem4(e4, d5)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = self.scem3(e3, d4)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = self.scem2(e2, d3)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        return 0, 0, 0, 0, d2  # To be consistent with mmseg


if __name__ == '__main__':
    model = SGBTransNet(224).cuda()

    img = torch.rand(4, 3, 224, 224).cuda()

    out = model(img)

    print(out[4].shape)
    print('debugger')
