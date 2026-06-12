# ResNet

## Architecture
ResNet is a CNN built from residual blocks. A residual block learns a correction term `F(x)` and adds it to the original input: `output = F(x) + x`
This skip connection allows the block to preserve spatial information while learning incremental refinements. Typical basic block:
```
Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> Add shortcut -> ReLU
```

## Problem
Very deep CNNs can suffer from the degradation problem -- adding more layers can make the training error worse, even when overfitting is not the issue. 
ResNet makes deep networks easier to optimize by allowing layers to learn residual functions instead of the full transformations. 

## Input / output
For image classification:
Input: `[B, C, H, W]`
Output: `[B, num_classes]`

For same-shape residual block:
Input: `[B, C, H, W]`
Output: `[B, C, H, W]`

For downsampling block:
Input: `[B, C_in, H, W]`
Output: `[B, C_out, H/stride, W/stride]`

If shape changes, shortcut path uses a projection: `1X1 conv -> BatchNorm`

## Key blocks

### BasicBlock
Used in deeper ResNets such as ResNet-18 and ResNet-34.
```
3x3 Conv -> BN -> ReLU -> 3x3 Conv -> BN -> Add shortcut -> ReLU
```

### BottleneckBlock
User in deeper ResNets like ResNet-50, ResNet-101 and ResNet-152.
```
1x1 Conv -> 3x3 conv -> 1x1 Conv
```
The `1x1` convolutions reduce and restore channel dimensions, making deeper models more efficient.

### Projection shortcut
Used when spatial size or channel count changes.
shortcut = 1x1 conv with stride -> BatchNorm

## Loss
For classification, ResNet commonly uses cross-entropy loss. The model outputs raw logits:
logits: `[B, num_classes]`
labels: `[B]`
Softmax is handled inside `CrossEntropyLoss`

## Why it was introduced
ResNet was introduced to make very deep CNNs easier to train. The key idea:
`If extra layers are not useful, a residual block can learn something close to identity`
This makes deeper networks less likely to perform worse simply because optimization became harder.

## Strengths
* Enables training very deep CNNs.
* Improves gradient flow through skip connections.
* Simple and modular design.
* Strong backbone for classification, detection, segmentation, and feature extraction.

## Weaknesses
* Limited global context compared to attention-based models.
* Can be expensive at large depth.
* Downsampling can lose spatial detail.
* Not naturally designed for sequence, multimodal, or long-range reasoning tasks.
* May be outperformed by ViT-style models when large-scale pretraining data is available.

## Production concerns
* Use `bias=False` for convolutions followed by BatchNorm.
* Use quantization, pruning, or distillation for deployment if needed.

## Important questions
1. Why do skip connections help?
They create an identity path for information and gradients. The block only needs to learn a residual correction instead of a full transformation.
2. Why use a 1x1 convolution in the shortcut?
A 1x1 convolution changes channel count and can downsample spatial size using stride, while keeping the shortcut path lightweight.
3. Why is padding used in 3x3 convolutions?
For kernel_size=3 and stride=1, padding=1 keeps height and width unchanged.
4. Why use Global Average Pooling before the classifier?
It converts spatial feature maps into one vector per image and reduces the number of parameters compared to flattening the full feature map.
5. How would ResNet be adapted for detection?
Use ResNet as a backbone to extract feature maps, often with a detection head such as Faster R-CNN, RetinaNet, or YOLO-style heads.
6. How would ResNet be adapted for segmentation?
Use ResNet as an encoder and add a decoder or segmentation head to produce dense pixel-level predictions.
7. How would you reduce ResNet inference latency?
Use a smaller variant, reduce input resolution, quantize weights, prune channels, use distillation, fuse Conv-BN layers, or deploy with optimized runtimes.
8. Why might ViT outperform ResNet?
ViTs can model long-range global interactions better and scale well with large datasets and pretraining.