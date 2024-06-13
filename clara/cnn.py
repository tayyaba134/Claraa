import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock1D3x3(nn.Module):
    def __init__(self, in_channel: int, out_channel: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channel, out_channel, 3, padding=1)
        self.conv2 = nn.Conv1d(out_channel, out_channel, 3, padding=1)
        self.bn = nn.BatchNorm1d(out_channel)

    def forward(self, x):
        x = F.gelu(self.conv1(x))
        return F.gelu(self.bn(self.conv2(x)))

class Cnn1D10(nn.Module):
    def __init__(self, n_mels: int, out_channels: int):
        super().__init__()
        self.layers = nn.ModuleList([
            ConvBlock1D3x3(in_size, out_size)
            for in_size, out_size in zip([n_mels] + [2048] * 4, [2048] * 4 + [out_channels])
        ])
        self.bidirectional_gru = nn.GRU(2048, 1024, num_layers=1, bidirectional=True, batch_first=True)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = F.dropout(x, p=0.2)
        # Permute for GRU input
        x = x.permute(0, 2, 1)  # Change (batch, channels, length) -> (batch, length, channels)
        x, _ = self.bidirectional_gru(x)
        return x

class Cnn1D12(Cnn1D10):
    # Inherits from Cnn1D10 and just uses different initial parameters
    def __init__(self, n_mels: int, out_channels: int):
        super().__init__(n_mels, out_channels)
        # Adjust the number of layers or other parameters as needed

#### 2D CNNs for Spectrogram Data with Bidirectional GRU
class ConvBlock2D3x3(nn.Module):
    def __init__(self, in_channel: int, out_channel: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = F.gelu(self.conv1(x))
        return F.gelu(self.bn(self.conv2(x)))

class Cnn2D10(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.ModuleList([
            ConvBlock2D3x3(in_size, out_size)
            for in_size, out_size in zip([in_channels] + [2048] * 4, [2048] * 4 + [out_channels])
        ])
        self.bidirectional_gru = nn.GRU(2048, 1024, num_layers=1, bidirectional=True, batch_first=True)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        # Assuming x shape is (batch, channels, height, width) and we process across width
        x = torch.mean(x, dim=2)  # Reduce across height
        x = x.permute(0, 2, 1)  # (batch, width, channels)
        x, _ = self.bidirectional_gru(x)
        return x
