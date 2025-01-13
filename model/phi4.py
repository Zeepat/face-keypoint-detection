# No time to try this out yet, but it seemed interesting.


import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class Hourglass(nn.Module):
    def __init__(self, num_blocks, in_channels, out_channels):
        super(Hourglass, self).__init__()
        self.residual_block1 = ResidualBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        layers = []
        for _ in range(num_blocks - 1):
            layers.append(ResidualBlock(64, 128))
        self.layers_down = nn.Sequential(*layers)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        layers = []
        for _ in range(num_blocks - 1):
            layers.append(ResidualBlock(128, 256))
        self.layers_up = nn.Sequential(*layers)

        self.residual_block3 = ResidualBlock(256, out_channels)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        # Downward path
        x1 = self.residual_block1(x)
        x2 = self.pool1(x1)
        x3 = self.layers_down(x2)
        x4 = self.pool2(x3)

        # Bottleneck
        x5 = self.upsample(x4)
        x6 = self.layers_up(x5)

        # Upward path
        x7 = self.residual_block3(x6 + x3)  # Skip connection

        return self.upsample(x7 + x1)  # Final output

class FacialLandmarkModel(nn.Module):
    def __init__(self, num_landmarks=68, num_stacks=2):
        super(FacialLandmarkModel, self).__init__()
        self.in_channels = 3
        self.num_blocks = 4
        self.num_stacks = num_stacks

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Stacks of hourglass modules
        self.hourglasses = nn.ModuleList([Hourglass(self.num_blocks, 64, num_landmarks) for _ in range(num_stacks)])
        
        # Output layers for each stack
        self.output_layers = nn.ModuleList([
            nn.Conv2d(64, num_landmarks * 2, kernel_size=1, stride=1)
            for _ in range(num_stacks)
        ])

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.pool(out)

        outputs = []
        for i in range(self.num_stacks):
            hg_out = self.hourglasses[i](out)
            output = self.output_layers[i](hg_out)
            outputs.append(output)
            if i < self.num_stacks - 1:
                # Intermediate supervision
                out = out + F.interpolate(output, scale_factor=2, mode='nearest')

        return outputs

# Example usage:
model = FacialLandmarkModel()
input_tensor = torch.randn(1, 3, 256, 256)  # Batch size of 1, 3 color channels, 256x256 image
output = model(input_tensor)
for i, out in enumerate(output):
    print(f"Stack {i} output shape: {out.shape}")