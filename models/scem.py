from models.sfrtrans import SFRTrans
from models.dpcca import DPCCA
import torch.nn as nn
import torch


class SCEM(nn.Module):
    def __init__(self, dpcca_ratio, **kwargs):
        super(SCEM, self).__init__()
        self.sfrtrans = SFRTrans(**kwargs)
        self.dpcca = DPCCA(kwargs['backbone_dim'], dpcca_ratio)

    def forward(self, E, D):
        S = self.sfrtrans(E, D)
        scem_output = self.dpcca(S, D)
        return scem_output


if __name__ == '__main__':
    scem = SCEM(2, backbone_dim=512, seq_dim=256, feature_map_size=28, patch_size=1, ffn_dim=256, head_dim=64).cuda()
    E = torch.rand(4, 512, 28, 28).cuda()
    D = torch.rand(4, 512, 28, 28).cuda()
    scem_output = scem(E, D)
    print(scem_output.shape)
    print('debugger')
