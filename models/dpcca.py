import torch.nn as nn
import torch


class DPCCA(nn.Module):
    """ channels: The number of channels of S and D.
        ratio: The ratio in channel-attention mechanism. """
    def __init__(self, channels, ratio=2):
        super(DPCCA, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.left_1 = nn.Sequential(
            nn.Linear(channels, channels // ratio),
            nn.ReLU(inplace=True)
        )
        self.right_1 = nn.Sequential(
            nn.Linear(channels, channels // ratio),
            nn.ReLU(inplace=True)
        )

        self.left_2 = nn.Sequential(
            nn.Linear(channels // ratio, channels),
            nn.Sigmoid()
        )
        self.right_2 = nn.Sequential(
            nn.Linear(channels // ratio, channels),
            nn.Sigmoid()
        )

    def forward(self, S, D):
        # S: The refined shallow feature maps from SFRTrans. D: The deep feature maps in decoder.
        left_embedding = self.gap(S).squeeze()
        right_embedding = self.gap(D).squeeze()

        left_embedding = left_embedding + right_embedding
        left_embedding = self.left_1(left_embedding)

        right_embedding = self.right_1(right_embedding)

        fusion_embedding = left_embedding + right_embedding

        left_embedding = self.left_2(fusion_embedding).unsqueeze(-1).unsqueeze(-1)
        right_embedding = self.right_2(fusion_embedding).unsqueeze(-1).unsqueeze(-1)

        return torch.concat([S * left_embedding, D * right_embedding], dim=1)


if __name__ == '__main__':

    _S = torch.rand(3, 64, 48, 48).cuda()
    _D = torch.rand(3, 64, 48, 48).cuda()

    dpcca = DPCCA(channels=64, ratio=2).cuda()
    output = dpcca(_S, _D)

    print(output.shape)
