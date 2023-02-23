import torch
from torch import nn
from einops import rearrange


class FeedForward(nn.Module):
    def __init__(self, input_dim, ffn_dim, dropout):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, input_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ffn(x)


class SelfAttention(nn.Module):
    def __init__(self, input_dim, head_dim, head_num, dropout):
        super().__init__()
        self.to_qkv = nn.Linear(input_dim, head_dim * head_num * 3, bias=False)
        self.scale = head_dim ** -0.5
        self.head_num = head_num
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Sequential(
            nn.Linear(head_dim * head_num, head_dim * head_num),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.head_num), qkv)
        attention_maps = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attention_maps = self.softmax(attention_maps)
        attention_maps = self.dropout(attention_maps)
        out = torch.matmul(attention_maps, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class SATransformer(nn.Module):
    def __init__(self, input_dim, ffn_dim, head_dim, head_num, dropout, depth):
        """ input_dim: the number of channels of the feature maps input to SATransformer, i.e., the
                number of channels of the feature maps output from CATransformer.
            ffn_dim: the intermediate dimension of the FFN.
            dropout: the dropout prob for SA and FFN. """
        super().__init__()
        self.sa_and_ffn = nn.ModuleList([])
        self.ln_list = nn.ModuleList([])
        for _ in range(depth):
            self.ln_list.append(nn.LayerNorm(head_dim * head_num))
            self.ln_list.append(nn.LayerNorm(head_dim * head_num))
            self.sa_and_ffn.append(
                nn.ModuleList([
                    SelfAttention(input_dim, head_dim=head_dim, head_num=head_num, dropout=dropout),
                    FeedForward(head_dim * head_num, ffn_dim, dropout=dropout)
                ])
            )

    def forward(self, x):
        i = 0
        for sa, ffn in self.sa_and_ffn:
            x = self.ln_list[i](sa(x) + x)
            x = self.ln_list[i+1](ffn(x) + x)
            i += 2
        return x


class CrossAttention(nn.Module):
    def __init__(self, seq_dim, head_dim, head_num, dropout):
        super().__init__()
        self.to_kv = nn.Linear(seq_dim, head_dim * head_num * 2, bias=False)
        self.to_q = nn.Linear(seq_dim, head_dim * head_num, bias=False)
        self.scale = head_dim ** -0.5
        self.head_num = head_num
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Sequential(
            nn.Linear(head_dim * head_num, head_dim * head_num),
            nn.Dropout(dropout)
        )

    def forward(self, seq_E, seq_D):
        kv = self.to_kv(seq_E).chunk(2, dim=-1)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.head_num), kv)

        q = self.to_q(seq_D)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.head_num)

        attention_maps = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attention_maps = self.softmax(attention_maps)
        attention_maps = self.dropout(attention_maps)
        out = torch.matmul(attention_maps, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class CATransformer(nn.Module):
    """CATransformer in SFRTrans.

    Args:
        seq_dim: the number of channels of the feature maps input to
            CATransformer, i.e., the dimension of seq_E and seq_D.
        ffn_dim: the intermediate dimension of the FFN.
        dropout: the dropout prob for CA and FFN.
    """

    def __init__(self, seq_dim, ffn_dim, head_dim, head_num, dropout):
        super().__init__()

        self.ca = CrossAttention(seq_dim, head_dim=head_dim, head_num=head_num, dropout=dropout)
        self.ln_after_ca = nn.LayerNorm(head_dim * head_num)

        self.ffn = FeedForward(head_dim * head_num, ffn_dim, dropout=dropout)
        self.ln_after_ffn = nn.LayerNorm(head_dim * head_num)

    def forward(self, seq_E, seq_D):
        x = self.ln_after_ca(self.ca(seq_E, seq_D) + seq_E)
        x = self.ln_after_ffn(self.ffn(x) + x)
        return x
