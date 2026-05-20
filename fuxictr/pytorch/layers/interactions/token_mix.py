import torch
from torch import nn

class MultiHeadTokenMixing(nn.Module):
    def __init__(self, input_dim, num_token):
        super(MultiHeadTokenMixing, self).__init__()
        self.num_token = num_token # = num_heads
        self.input_dim = input_dim
        assert input_dim % num_token == 0, "input_dim must be divisible by num_tokens"
        self.head_dim = self.input_dim // self.num_token

    def forward(self, x):  # x: [B, T, D]
        heads = torch.tensor_split(x, self.num_token, dim=-1)  # list(H) of [B, T, Dh]
        mixed = torch.stack(heads, dim=1)                      # [B, H, T, Dh]
        out = mixed.flatten(start_dim=2)                     # [B, H, T*Dh]，当 H=T, T*Dh=D -> [B,T,D]
        return out
