import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNeXtBottleneck(nn.Module):
    """
    ResNeXt bottleneck type C
    """
    def __init__(self, in_channels, out_channels, stride, cardinality, base_width, widen_factor):
        super().__init__()
        width_ratio = out_channels / (widen_factor * 64.)
        D = cardinality * int(base_width * width_ratio)
        self.conv_reduce = nn.Conv2d(in_channels, D, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_reduce = nn.BatchNorm2d(D)
        self.conv_conv = nn.Conv2d(D, D, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn = nn.BatchNorm2d(D)
        self.conv_expand = nn.Conv2d(D, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_expand = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.leaky_relu(self.bn_reduce(self.conv_reduce(x)), inplace=True)
        x = F.leaky_relu(self.bn(self.conv_conv(x)), inplace=True)
        x = self.bn_expand(self.conv_expand(x))
        return F.leaky_relu(x + residual, inplace=True)

class BidirectionalResNeXt(nn.Module):
    """
    ResNext optimized for bidirectional processing.
    """
    def __init__(self, cardinality, depth, num_classes, base_width, widen_factor=4):
        super().__init__()
        self.cardinality = cardinality
        self.depth = depth
        self.block_depth = (self.depth - 2) // 9
        self.base_width = base_width
        self.widen_factor = widen_factor
        self.num_classes = num_classes
        self.stages = [64, 64 * widen_factor, 128 * widen_factor, 256 * widen_factor]

        self.conv_1_3x3 = nn.Conv2d(1, 64, 3, 1, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, self.stages[0], 1)
        self.layer2 = self._make_layer(self.stages[0], self.stages[1], 2)
        self.layer3 = self._make_layer(self.stages[1], self.stages[2], 2)
        self.layer4 = self._make_layer(self.stages[2], self.stages[3], 2)

        self.fc = nn.Linear(self.stages[3] * 2, num_classes)  # Combining features from both directions

    def _make_layer(self, in_channels, out_channels, stride):
        return ResNeXtBottleneck(in_channels, out_channels, stride, self.cardinality, self.base_width, self.widen_factor)

    def forward(self, x):
        # Forward stream
        x1 = self.conv_1_3x3(x)
        x1 = F.leaky_relu(self.bn_1(x1), inplace=True)
        x1 = self.layer1(x1)
        x1 = self.layer2(x1)
        x1 = self.layer3(x1)
        x1 = self.layer4(x1)
        x1 = x1.mean([-2, -1])  # Global pooling

        # Reverse stream
        x2 = torch.flip(x, dims=[-1])  # Assuming time or sequence is the last dimension
        x2 = self.conv_1_3x3(x2)
        x2 = F.leaky_relu(self.bn_1(x2), inplace=True)
        x2 = self.layer1(x2)
        x2 = self.layer2(x2)
        x2 = self.layer3(x2)
        x2 = self.layer4(x2)
        x2 = x2.mean([-2, -1])  # Global pooling

        # Combining both directional features
        x_combined = torch.cat([x1, x2], dim=1)
        out = self.fc(x_combined)

        return out

# Example Usage
#model = BidirectionalResNeXt(cardinality=32, depth=29, num_classes=10, base_width=4, widen_factor=4)
#input_tensor = torch.rand(1, 1, 224, 224)  # Example input tensor
#output = model(input_tensor)
