"""
Convolutional block attention module (CBAM).
"""


import torch
from torch import nn
from torch.nn import functional as F


class CBAMChannelAttention(nn.Module):
	"""
	CBAM's channel attention module.

	Args:
		in_dim (int): Number of input channels.
		reduction_factor (int): Reduction factor for the 
		bottleneck layer.
		Default is 16.
	"""
	def __init__(
		self, 
		in_dim: int,
		reduction_factor: int = 16,
		) -> None:
		super().__init__()

		bottleneck_dim = in_dim//reduction_factor
		self.mlp = nn.Sequential(
			nn.Conv2d(
				in_channels=in_dim,
				out_channels=bottleneck_dim,
				kernel_size=1,
				),
			nn.ReLU(),
			nn.Conv2d(
				in_channels=bottleneck_dim,
				out_channels=in_dim,
				kernel_size=1,
				),
			)

	def forward(self, input: torch.Tensor) -> torch.Tensor:
		avg_pooled = F.adaptive_avg_pool2d(input, 1)
		max_pooled = F.adaptive_avg_pool2d(input, 1)

		avg_attention = self.mlp(avg_pooled)
		max_attention = self.mlp(max_pooled)
		
		attention = avg_attention+max_attention
		attention = F.sigmoid(attention)
		
		output = attention*input
		return output


def channel_avg_pool(input: torch.Tensor) -> torch.Tensor:
	"""
	Average pool along the channel axis.

	Args:
		input (torch.Tensor): Input to average pool.

	Returns (torch.Tensor): Input average pooled over the channel axis.
	"""
	return input.mean(dim=1, keepdim=True)


def channel_max_pool(input: torch.Tensor) -> torch.Tensor:
	"""
	Max pool along the channel axis.

	Args:
		input (torch.Tensor): Input to max pool.

	Returns (torch.Tensor): Input max pooled over the channel axis.
	"""
	return input.max(dim=1, keepdim=True).values


class CBAMSpatialAttention(nn.Module):
	"""
	CBAM's spatial attention.

	Args:
		kernel_size (int): Kernel size of the convolution.
		Default is 7.
	"""
	def __init__(
		self, 
		kernel_size: int = 7,
		) -> None:
		super().__init__()

		self.conv = nn.Conv2d(
			in_channels=2,
			out_channels=1,
			kernel_size=kernel_size,
			padding=kernel_size//2,
			)
	
	def forward(self, input: torch.Tensor) -> torch.Tensor:
		avg_pooled = channel_avg_pool(input)
		max_pooled = channel_max_pool(input)
		pooled = torch.cat([avg_pooled, max_pooled], dim=1)

		attention = self.conv(pooled)
		attention = F.sigmoid(attention)		
		
		output = attention*input
		return output


class CBAM(nn.Module):
	"""
	Convolutional block attention module.

	Args:
		in_dim (int): Number of input channels.
		reduction_factor (int): Reduction factor for the 
		bottleneck layer of the channel attention module.
		Default is 16.
		kernel_size (int): Kernel size for the convolution 
		of the spatial attention module.
		Default is 7.
	"""
	def __init__(
		self, 
		in_dim: int,
		reduction_factor: int = 16,
		kernel_size: int = 7,
		) -> None:
		super().__init__()

		self.channel_attention = CBAMChannelAttention(
			in_dim=in_dim,
			reduction_factor=reduction_factor,
			)
		self.spatial_attention = CBAMSpatialAttention(kernel_size)
	
	def forward(self, input: torch.Tensor) -> torch.Tensor: 
		output = self.channel_attention(input)
		output = self.spatial_attention(output)
		return output
