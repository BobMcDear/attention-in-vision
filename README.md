# Attention in Vision
## Description
This is a PyTorch implementation of popular attention mechanisms in visions. You can find the accompanying blog series [here](https://borna-ahz.medium.com/attention-in-computer-vision-part-1-se-ese-and-eca-c5effac7c11e).
## Modules
Below is a list of the attention mechanisms available in the repository, as well as their sample usage. For more information, please refer to the corresponding paper or article of each method.
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
Squeeze-and-excitation (SE) is accessed as follows.
```python
from eca import ECA


eca = ECA(
        kernel_size=kernel_size, # Neighbourhood size, i.e., kernel size of the 1D convolution
        )
output = eca(input)
```
