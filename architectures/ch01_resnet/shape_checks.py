import torch

from resnet_blocks import BasicBlock
from mini_resnet import MiniResNet


def check_same_shape_block():
    x = torch.randn(4, 64, 32, 32)
    block = BasicBlock(in_channels=64, out_channels=64, stride=1)
    y = block(x)
    print("check_same_shape_block: Shapes y, x = ", y.shape, x.shape)
    assert(y.shape == x.shape) # torch.Size([4, 64, 32, 32]) torch.Size([4, 64, 32, 32])

def check_downsample_block():
    x = torch.randn(4, 64, 32, 32)
    block = BasicBlock(in_channels=64, out_channels=128, stride=2)
    y = block(x)
    print("check_downsample_block: Shapes y, x = ", y.shape, x.shape)
    assert(y.shape != x.shape) # torch.Size([4, 128, 16, 16]), torch.Size([4, 64, 32, 32])

def check_channel_change_only():
    x = torch.randn(4, 64, 32, 32)
    block = BasicBlock(in_channels=64, out_channels=128, stride=1)
    y = block(x)
    print("check_channel_change_only: Shapes y, x = ", y.shape, x.shape)
    assert(y.shape != x.shape) # torch.Size([4, 128, 32, 32]), torch.Size([4, 64, 32, 32])

def check_mini_resnet():
    x = torch.randn(4, 3, 32, 32)
    model = MiniResNet(num_classes=10)
    y = model(x)
    print("check_mini_resnet: ", "input: ", x.shape, ", output:", y.shape)
    assert y.shape == torch.Size([4, 10])


def main():
    check_same_shape_block()
    check_downsample_block()
    check_channel_change_only()
    check_mini_resnet()
    print("All ResNet BasicBlock shape checks passed.")

if __name__ == '__main__':
    main()
