import torch.nn as nn


def depth_conv(in_channels, stride_value=1):
    return (
        nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=stride_value, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
        )
    )

def pointwise_conv(in_channels, out_channels):
    return (
        nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )
    )

def standard_conv(in_channels, out_channels, stride_value):
    return (
        nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride_value, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )
    )

class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_ratio, stride_value):
        super(InvertedResidualBlock, self).__init__()

        self.stride = stride_value
        assert stride_value in [1, 2]

        hidden_channels = in_channels * expansion_ratio

        self.residual_connection = self.stride == 1 and in_channels == out_channels

        block_layers = []
        if expansion_ratio != 1:
            block_layers.append(pointwise_conv(in_channels, hidden_channels))
        block_layers.extend([
            depth_conv(hidden_channels, stride_value=stride_value),
            pointwise_conv(hidden_channels, out_channels)
        ])

        self.block_layers = nn.Sequential(*block_layers)

    def forward(self, x):
        if self.residual_connection:
            return x + self.block_layers(x)
        else:
            return self.block_layers(x)

class MobileNetV2(nn.Module):    
    def __init__(self, input_channels=3, num_classes=10, compute_device='cpu'):
        super(MobileNetV2, self).__init__()
        self.compute_device = compute_device

        self.block_configs = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1]
        ]

        self.initial_conv = standard_conv(input_channels, 32, stride_value=2)
      
        blocks = []
        current_channel = 32
        for t, c, n, s in self.block_configs:
            for i in range(n):
                stride = s if i == 0 else 1
                blocks.append(InvertedResidualBlock(in_channels=current_channel, out_channels=c, expansion_ratio=t, stride_value=stride))
                current_channel = c

        self.blocks = nn.Sequential(*blocks)

        self.final_conv = pointwise_conv(current_channel, 1280)

        self.network_classifier = nn.Sequential(
            nn.Dropout2d(0.2),
            nn.Linear(1280, num_classes)
        )
        self.pool_layer = nn.AdaptiveAvgPool2d(1)
        # self.to(compute_device)

    def forward(self, x):
        # x = x.to(self.compute_device)
        x = self.initial_conv(x)
        x = self.blocks(x)
        x = self.final_conv(x)
        x = self.pool_layer(x).view(-1, 1280)
        x = self.network_classifier(x)
        return x


