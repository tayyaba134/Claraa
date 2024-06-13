import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
from typing import Union, Callable, Tuple, Optional

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)

class PositionalEncoding(nn.Module):
    """ Positional Encoding using sine and cosine functions. """
    def __init__(self, in_channels: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, in_channels, 2) * (-math.log(10000.0) / in_channels))
        pe = torch.zeros(max_len, 1, in_channels)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerEncoderLayerBidirectional(nn.Module):
    """ Transformer Encoder layer processing both directions. """
    def __init__(self, in_channels: int, out_channels: int, num_layers: int, nhead: int, dim_feedforward: int=1024,
                 dropout: float=0.1, activation: Union[str, Callable[[Tensor], Tensor]] = F.relu, 
                 layer_norm_eps: float=1e-5, batch_first: bool=False, norm_first: bool=False):
        super().__init__()
        self.forward_transformer = nn.TransformerEncoderLayer(
            d_model=in_channels, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, 
            activation=activation, layer_norm_eps=layer_norm_eps, batch_first=batch_first, norm_first=norm_first
        )
        self.reverse_transformer = nn.TransformerEncoderLayer(
            d_model=in_channels, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, 
            activation=activation, layer_norm_eps=layer_norm_eps, batch_first=batch_first, norm_first=norm_first
        )
        self.norm = LayerNorm(in_channels)
        self.project = nn.Linear(in_channels, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        x_orig = self.forward_transformer(x)
        x_rev = self.reverse_transformer(torch.flip(x, dims=[1]))  # Assuming time dimension is 1
        x_combined = self.norm(x_orig + x_rev)
        return self.project(x_combined)

class BidirectionalCLARAEncoder(nn.Module):
    """ Bidirectional Encoder for CLARA. """
    def __init__(self, in_channels: int, out_channels: int, num_layers: int, nhead: int, dim_feedforward: int=1024,
                 dropout: float=0.1, activation: Union[str, Callable[[Tensor], Tensor]] = F.relu, 
                 layer_norm_eps: float=1e-5, batch_first: bool=False):
        super().__init__()
        self.positional_encoding = PositionalEncoding(in_channels, dropout)
        self.encoder = TransformerEncoderLayerBidirectional(
            in_channels, out_channels, num_layers, nhead, dim_feedforward, dropout, activation, layer_norm_eps, batch_first
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.positional_encoding(x)
        return self.encoder(x)

# Example usage
#model = BidirectionalCLARAEncoder(in_channels=512, out_channels=512, num_layers=6, nhead=8)
#input_tensor = torch.rand(10, 20, 512)  # Example tensor with shape [batch, seq_length, features]
#output = model(input_tensor)
