import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from typing import Union, Callable

from modules import TransformerEncoderLayerBidirectional, LayerNorm

class BidirectionalSimpleTransformer(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_layers: int,
                 nhead: int,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 0.00001,
                 batch_first: bool = True,  # Assuming True for handling sequence data
                 norm_first: bool = False
                 ):
        super().__init__()
        self.forward_transformer_encoder = TransformerEncoderLayerBidirectional(
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first
        )
        self.reverse_transformer_encoder = TransformerEncoderLayerBidirectional(
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first
        )
        self.ln = LayerNorm(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        # Process the original sequence
        x_orig = self.forward_transformer_encoder(x)

        # Reverse the sequence and process
        x_rev = torch.flip(x, dims=[1])  # Flip over sequence length dimension
        x_rev = self.reverse_transformer_encoder(x_rev)

        # Combine the results
        x_combined = x_orig + x_rev  # Element-wise addition
        return self.ln(x_combined)

# Example usage
#model = BidirectionalSimpleTransformer(
 #   in_channels=512, out_channels=512, num_layers=6, nhead=8
#)
#input_tensor = torch.rand(10, 20, 512)  # Example tensor with shape [batch, seq_length, features]
#output = model(input_tensor)
