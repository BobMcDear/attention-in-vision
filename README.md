# Attention in Vision

• <strong>[Introduction](#introduction)</strong><br>
• <strong>[Modules](#modules)</strong><br>
&nbsp;&nbsp;&nbsp; • <strong>[Squeeze-and-Excitation (SE)](#squeeze-and-excitation-se--paper)</strong><br>
&nbsp;&nbsp;&nbsp; • <strong>[Effective Squeeze-and-Excitation (eSE)](#effective-squeeze-and-excitation-ese--paper)</strong><br>
&nbsp;&nbsp;&nbsp; • <strong>[Efficient Channel Attention (ECA)](#efficient-channel-attention-eca--paper)</strong><br>
&nbsp;&nbsp;&nbsp; • <strong>[Convolutional Block Attention Module (CBAM)](#convolutional-block-attention-module-cbam--paper)</strong><br>
&nbsp;&nbsp;&nbsp; • <strong>[Bottleneck Attention Module (BAM)](#bottleneck-attention-module-bam--paper)</strong><br>
&nbsp;&nbsp;&nbsp; • <strong>[Gather-Excite (GE)](#gather-excite-ge--paper)</strong><br>
&nbsp;&nbsp;&nbsp; • <strong>[Selective Kernel (SK)](#selective-kernel-sk--paper)</strong><br>
&nbsp;&nbsp;&nbsp; • <strong>[Split Attention (SplAt)](#split-attention-splat--paper)</strong><br>
&nbsp;&nbsp;&nbsp; • <strong>[Conditionally Parameterized Convolution (CondConv)](#conditionally-parameterized-convolution-condconv--paper)</strong><br>
&nbsp;&nbsp;&nbsp; • <strong>[Dynamic convolution](#dynamic-convolution--paper)</strong><br>
&nbsp;&nbsp;&nbsp; • <strong>[Multi-Headed Self-Attention (MHSA)](#multi-headed-self-attention-mhsa--paper)</strong><br>

## Introduction
PyTorch implementations of popular attention mechanisms in computer vision can be found in this repository. The code aims to be lean, usable out of the box, and efficient, but first and foremost readable and instructive for those seeking to explore attention modules in vision. This repository is also accompanied by a [blog post](https://bobmcdear.github.io/posts/attention-in-vision/) that studies each layer in detail and elaborates on the code, so the two should be considered complementary material.
## Modules
Below is a list of available attention layers, as well as their sample usage.

### Squeeze-and-Excitation (SE) | [Paper](https://arxiv.org/abs/1709.01507)
Squeeze-and-excitation (SE) is accessed as follows.

```python
from se import SE

se = SE(
        in_dim=in_dim, # Number of channels SE receives
        reduction_factor=reduction_factor, # Reduction factor for the excitation module
        )
```
### Effective Squeeze-and-Excitation (eSE) | [Paper](https://arxiv.org/abs/1911.06667)
Effective squeeze-and-excitation (eSE) is accessed as follows.
```python
from ese import eSE


ese = eSE(
        in_dim=in_dim, # Number of channels eSE receives
        )
```
### Efficient Channel Attention (ECA) | [Paper](https://arxiv.org/abs/1910.03151)
Efficient channel attention (ECA) is accessed as follows.
```python
from eca import ECA


# ECA with automatically-calculated kernel size
eca = ECA(
        beta=beta, # beta value used in calculating the kernel size
        gamma=gamma, # gamma value used in calculating the kernel size
        in_dim=in_dim, # Number of channels ECA receives, required when kernel size is None
        )
# ECA with custom kernel size
eca = ECA(
        kernel_size=kernel_size, # Neighbourhood size, i.e., kernel size of the 1D convolution
        )
```

### Convolutional Block Attention Module (CBAM) | [Paper](https://arxiv.org/abs/1807.06521)
Convolutional block attention module (CBAM) is accessed as follows.
```python
from cbam import CBAM


cbam = CBAM(
        in_dim=in_dim, # Number of channels CBAM receives
        reduction_factor=reduction_factor, # Reduction factor for channel attention
        kernel_size=kernel_size, # Kernel size for spatial attention
        )
```

### Bottleneck Attention Module (BAM) | [Paper](https://arxiv.org/abs/1807.06514)
Bottleneck attention module (BAM) is accessed as follows.
```python
from bam import BAM


bam = BAM(
        in_dim=in_dim, # Number of channels BAM receives
        reduction_factor=reduction_factor, # Reduction factor for channel and spatial attention
        kernel_size=kernel_size, # Dilation for spatial attention
        )
```

### Gather-Excite (GE) | [Paper](https://arxiv.org/abs/1810.12348)
Gather-excite (GE) is accessed as follows.
```python
from bam import BAM

# GE-θ-
ge_no_params = GENoParams(
        extent=extent, # Extent factor, 0 for a global extent
        )
# GE-θ
ge_params = GEParams(
        in_dim=in_dim, # Number of channels GE receives
        extent=extent, # Extent factor, 0 for a global extent
        spatial_dim=spatial_dim, # Spatial dimension GE receives, required for a global extent
        )
# GE-θ+
ge_params_plus = GEParamsPlus(
        in_dim=in_dim, # Number of channels GE receives
        extent=extent, # Extent factor, 0 for a global extent
        spatial_dim=spatial_dim, # Spatial dimension GE receives, required for a global extent
        )
```

### Selective Kernel (SK) | [Paper](https://arxiv.org/abs/1903.06586)
Selective kernel module (SK) is accessed as follows.
```python
from sk import SK

sk = SK(
        in_dim=in_dim, # Number of channels SK receives
        out_dim=out_dim, # Desired number of output channels
        n_branches=n_branches, # Number of branches
        stride=stride, # Stride of each branch
        groups=groups, # Number of groups per branch
        reduction_factor=reduction_factor, # Reduction factor for the MLP calculating attention values
        )
```

### Split Attention (SplAt) | [Paper](https://arxiv.org/abs/2004.08955)
Split attention (SplAt) is accessed as follows.
```python
from splat import SplAt

splat = SplAt(
        in_dim=in_dim, # Number of channels SplAt receives
        out_dim=out_dim, # Desired number of output channels
        kernel_size=kernel_size, # Kernel size of SplAt
        stride=stride, # Stride of SplAt
        cardinality=cardinality, # Number of cardinal groups
        radix=radix, # Radix
        reduction_factor=reduction_factor, # # Reduction factor for the MLP calculating attention values
        )
```

### Conditionally Parameterized Convolution (CondConv) | [Paper](https://arxiv.org/abs/1904.04971)
Conditionally Parameterized Convolution (CondConv) is accessed as follows.
```python
from condconv import CondConv

condconv = CondConv(
        in_dim=in_dim, # Number of channels CondConv receives
        out_dim=out_dim, # Desired number of output channels
        kernel_size=kernel_size, # Kernel size of each expert/convolultion
        stride=stride, # Stride of each expert/convolultion
        padding=padding, # Padding of each expert/convolution
        dilation=dilation, # Dilation of each expert/convolution
        groups=groups, # Number of groups per expert/convolultion
        bias=bias, # Whether the experts/convolutions should contain bias terms
        n_experts=n_experts, # Number of experts/convolutions
        )
```

### Dynamic Convolution | [Paper](https://arxiv.org/abs/1912.03458)
Dynamic convolution is accessed as follows.
```python
from dynamic_conv import DynamicConv

dynamic_conv = DynamicConv(
        in_dim=in_dim, # Number of channels dynamic convolution receives
        out_dim=out_dim, # Desired number of output channels
        kernel_size=kernel_size, # Kernel size of each expert/convolultion
        stride=stride, # Stride of each expert/convolultion
        padding=padding, # Padding of each expert/convolution
        dilation=dilation, # Dilation of each expert/convolution
        groups=groups, # Number of groups per expert/convolultion
        bias=bias, # Whether the experts/convolutions should contain bias terms
        n_experts=n_experts, # Number of experts/convolutions
        reduction_factor=reduction_factor, # Reduction factor in the MLP of the router
        temperature=temperature, # Temperature coefficient for softmax in the router
        )
```

### Multi-Headed Self-Attention (MHSA) | [Paper](https://arxiv.org/abs/1706.03762)
Multi-headed self-attention is accessed as follows.
```python
from mhsa import MHSA

# Input should be of shape [batch_size, n_tokens, token_dim]
mhsa = MHSA(
        in_dim=in_dim, # Dimension of input, i.e., embedding or token dimension
        n_heads=n_heads, # Number of heads
        )
```
