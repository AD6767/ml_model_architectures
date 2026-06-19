import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """
    Basic residual block for same input / output shape. Also includes optional projection shortcut.

    if stride = 1, in_channels = out_channels
        Input: [B, C, H, W]
        Output: [B, C, H, W]
    if stride = 2, in_channels != out_channels
        Input: [B, C_in, H, W]
        Output: [B, C_out, H/2, W/2]
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        # Basic block
        # 1. conv1: 3x3 convolution, channels -> channels
        # 2. bn1
        # 3. relu
        # 4. conv2: 3x3 convolution, channels -> channels
        # 5. bn2
        # ------
        # Generalized version
        # conv1 should use stride=stride
        # conv2 should use stride=1
        # define self.shortcut
        # If shape changes, use projection:
        #   nn.Sequential(1x1 conv, BatchNorm2d)
        # Else:
        #   nn.Identity()
        # Because each conv is followed by BatchNorm, the conv bias is usually unnecessary. BatchNorm already has learnable shift/scale parameters.
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,3), stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3,3), stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1,1), stride=stride, padding=0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Basic block
        # 1. save identity = x
        # 2. residual path: conv1 -> bn1 -> relu -> conv2 -> bn2
        # 3. add identity
        # The residual branch produces a correction F(x), then we add it to the identity path to produce F(x) + x. 
        # The final ReLU is applied after combining both paths. If we applied ReLU too early, we would restrict the 
        # residual branch before it can contribute positive or negative corrections to the identity.
        # 4. final relu
        # 5. return output
        # ------ Generalized version
        # identity = self.shortcut(x)
        # residual path through conv1/bn1/relu/conv2/bn2
        # add residual + identity
        # final relu
        identity_x = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        f_x = self.bn2(self.conv2(x))
        h_x = f_x + identity_x
        out = self.relu(h_x)
        return out
