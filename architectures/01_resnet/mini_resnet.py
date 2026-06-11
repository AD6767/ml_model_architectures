import torch
import torch.nn as nn

from resnet_blocks import BasicBlock

'''
Small ResNet Classifier flow --
image
 -> stem conv
 -> residual stages
 -> global average pooling
 -> linear classifier
 -> logits
'''
class MiniResNet(nn.Module):
    """
    Tiny ResNet-style classifier for CIFAR-10 like images
    Input: [B, 3, 32, 32]
    Output: [B, num_classes]
    """
    
    def __init__(self, num_classes: int = 10):
        super().__init__()

        # 1. stem conv (channels: 3 -> 64) (input shape = output shape)
        conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), stride=1, padding=1, bias=False) # Since the stem conv is followed by BatchNorm, bias false.
        bn2d = nn.BatchNorm2d(num_features=64)
        relu = nn.ReLU()
        self.stem = nn.Sequential(conv1, bn2d, relu) # The stem conv converts raw pixels into low-level features: edges, corners, textures, color patterns. 
                                                     # We keep the spatial shape because CIFAR images are already small. If we downsample too early, we lose too much information.

        # 2. Residual Stage -- Keep shape
        self.stage1 = BasicBlock(in_channels=64, out_channels=64, stride=1) # keep shape here Because early layers need to capture small visual details.
                                                                            # For small images, downsampling immediately can destroy information.

        # 3. Residual Stage -- Downsample [B, 128, 16, 16]
        self.stage2 = BasicBlock(in_channels=64, out_channels=128, stride=2) # At deeper layers, we care less about exact pixel-level detail and more about higher-level patterns. 
                                                                             # Increasing channels gives the model more capacity to represent richer features.

        # 4. Residual Stage -- Downsample [B, 256, 8, 8]
        self.stage3 = BasicBlock(in_channels=128, out_channels=256, stride=2) # Intuition -- early:  where are edges/textures?, 
                                                                              # middle: where are object parts?, 
                                                                              # late: what object/category is present?

        # 5. Global Average Pooling -> [B, 256, 1, 1]. Converts spatial feature maps into one vector per image.
        # self.avg_pool = nn.AvgPool2d(kernel_size=(8,8)) # assumes the feature map is always 8x8.
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1)) # Whatever spatial size comes in, convert it to [1, 1].
                                            
        # 6. Linear Classifier -> [B, num_classes]
        self.fc = nn.Linear(in_features=256, out_features=num_classes)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x) # [B, 3, 32, 32] -> [B, 64, 32, 32]
        x = self.stage1(x) # [B, 64, 32, 32] -> [B, 64, 32, 32]
        x = self.stage2(x) # [B, 64, 32, 32] -> [B, 128, 16, 16]
        x = self.stage3(x) # [B, 128, 16, 16] -> [B, 256, 8, 8]
        x = self.avg_pool(x) # [B, 256, 8, 8] -> [B, 256, 1, 1]
        # dim 0 = batch, dim 1 = channels, dim 2 = height, dim 3 = width
        x = torch.flatten(x, start_dim=1) # [B, 256, 1, 1] -> [B, 256] (start flattening from dim 1 onwards; 256 * 1 * 1 = 256)
        # x = x.view(x.size(0), -1) --> x.view(B, -1) --> [B, 256] or x.reshape(x.size(0), -1)
        out = self.fc(x) # [B, 256] -> [B, num_classes]
        return out
