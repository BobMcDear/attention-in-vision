# Attention in Vision
## Description
This is a PyTorch implementation of popular attention mechanisms in computer vision. You can find the accompanying blog series [here](https://borna-ahz.medium.com/attention-in-computer-vision-part-1-se-ese-and-eca-c5effac7c11e).
## Modules
Below is a list of the attention mechanisms available in this repository, as well as their sample usage. For more information, please refer to the paper or article corresponding to each method.
### SE | [Paper](https://arxiv.org/abs/1709.01507) - [Blog](https://borna-ahz.medium.com/attention-in-computer-vision-part-1-se-ese-and-eca-c5effac7c11e)
Squeeze-and-excitation (SE) is accessed as follows.
```python
from se import SE


se = SE(
        in_dim=in_dim, # Number of channels SE receives
        reduction_factor=reduction_factor, # Reduction factor for the excitation module
        )
output = se(input)
```
### ECA | [Paper](https://arxiv.org/abs/1910.03151) - [Blog](https://borna-ahz.medium.com/attention-in-computer-vision-part-1-se-ese-and-eca-c5effac7c11e)
Efficient channel attention (ECA) is accessed as follows.
```python
from eca import ECA


eca = ECA(
        kernel_size=kernel_size, # Neighbourhood size, i.e., kernel size of the 1D convolution
        )
output = eca(input)
```

### CBAM | [Paper](https://arxiv.org/abs/1807.06521) - [Blog](https://borna-ahz.medium.com/attention-in-computer-vision-part-2-cbam-and-bam-e482112a26db)
Convolutional block attention module (CBAM) is accessed as follows.
```python
from cbam import CBAM


cbam = CBAM(
        in_dim=in_dim, # Number of channels CBAM receives
        reduction_factor=reduction_factor, # Reduction factor for channel attention
        kernel_size=kernel_size, # Kernel size for spatial attention
        )
output = cbam(input)
```

### BAM | [Paper](https://arxiv.org/abs/1807.06514) - [Blog](https://borna-ahz.medium.com/attention-in-computer-vision-part-2-cbam-and-bam-e482112a26db)
Bottleneck attention module (BAM) is accessed as follows.
```python
from bam import BAM


bam = BAM(
        in_dim=in_dim, # Number of channels BAM receives
        reduction_factor=reduction_factor, # Reduction factor for channel and spatial attention
        kernel_size=kernel_size, # Dilation for spatial attention
        )
output = bam(input)
```

