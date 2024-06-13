import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import PositionalEncoding

class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)

class Conv1d(nn.Conv1d):
    def _conv_forward(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        return super()._conv_forward(x, weight.to(x.dtype), bias.to(x.dtype) if bias is not None else None)

class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.d_k = self.d_v = n_state // n_head

        self.query = nn.Linear(n_state, n_head * self.d_k)
        self.key = nn.Linear(n_state, n_head * self.d_k)
        self.value = nn.Linear(n_state, n_head * self.d_v)
        self.out = nn.Linear(n_head * self.d_v, n_state)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        q = self.query(q).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        k = self.key(k).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        v = self.value(v).view(batch_size, -1, self.n_head, self.d_v).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / self.d_k**0.5
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        output = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch_size, -1, self.n_head * self.d_v)
        return self.out(output)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)
        self.mlp = nn.Sequential(
            nn.Linear(n_state, n_state * 4),
            nn.GELU(),
            nn.Linear(n_state * 4, n_state),
        )
        self.mlp_ln = LayerNorm(n_state)

    def forward(self, x):
        q = k = v = x
        x = x + self.attn(self.attn_ln(x), q, k, v)
        x = x + self.mlp(self.mlp_ln(x))
        return x

class BidirectionalWhisperAudioEncoder(nn.Module):
    def __init__(self, n_mels: int, n_state: int, n_head: int, n_layer: int):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.positional_embedding = PositionalEncoding(n_state)
        self.blocks = nn.ModuleList([ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)])
        self.ln_post = LayerNorm(n_state)

    def forward(self, x):
        x_orig = self.process_sequence(x)
        x_rev = self.process_sequence(torch.flip(x, dims=[-1]))
        x_combined = x_orig + x_rev
        return x_combined

    def process_sequence(self, x):
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)
        x = self.positional_embedding(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_post(x)
        x = x.permute(0, 2, 1)
        return x

# Example Usage
#model = BidirectionalWhisperAudioEncoder(n_mels=80, n_state=512, n_head=8, n_layer=6)
#input_tensor = torch.rand(1, 80, 400)  # Example input tensor
#output = model(input_tensor)
