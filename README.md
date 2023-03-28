# Attention in Vision
## Description
This is a PyTorch implementation of popular attention mechanisms in computer vision.
## Modules
Below is a list of the attention mechanisms available in this repository, as well as their sample usage.

### SE | [Paper](https://arxiv.org/abs/1709.01507)
Squeeze-and-excitation (SE) is accessed as follows.
```python
from se import SE


se = SE(
        in_dim=in_dim, # Number of channels SE receives
        reduction_factor=reduction_factor, # Reduction factor for the excitation module
        )
```
### eSE | [Paper](https://arxiv.org/abs/1911.06667)
Effective squeeze-and-excitation (eSE) is accessed as follows.
```python
from ese import eSE


ese = eSE(
        in_dim=in_dim, # Number of channels eSE receives
        )
```
### ECA | [Paper](https://arxiv.org/abs/1910.03151)
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

### CBAM | [Paper](https://arxiv.org/abs/1807.06521)
Convolutional block attention module (CBAM) is accessed as follows.
```python
from cbam import CBAM


cbam = CBAM(
        in_dim=in_dim, # Number of channels CBAM receives
        reduction_factor=reduction_factor, # Reduction factor for channel attention
        kernel_size=kernel_size, # Kernel size for spatial attention
        )
```

### BAM | [Paper](https://arxiv.org/abs/1807.06514)
Bottleneck attention module (BAM) is accessed as follows.
```python
from bam import BAM


bam = BAM(
        in_dim=in_dim, # Number of channels BAM receives
        reduction_factor=reduction_factor, # Reduction factor for channel and spatial attention
        kernel_size=kernel_size, # Dilation for spatial attention
        )
```

### GE | [Paper](https://arxiv.org/abs/1810.12348)
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
