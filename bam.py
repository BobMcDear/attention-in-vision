"""
Bottleneck attention module (BAM).
"""


import torch
from torch import nn
from torch.nn import functional as F


class BAMChannelAttention(nn.Module):
	"""
	BAM's channel attention module.

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
			nn.BatchNorm2d(bottleneck_dim),
			nn.ReLU(),
			nn.Conv2d(
				in_channels=bottleneck_dim,
				out_channels=in_dim,
				kernel_size=1,
				),
			)
	
	def forward(self, input: torch.Tensor) -> torch.Tensor:
		avg_pooled = F.adaptive_avg_pool2d(input, 1)
		attention = self.mlp(avg_pooled)
		return attention


class BAMSpatialAttention(nn.Module):
	"""
	BAM's spatial attention module.

	Args:
		in_dim (int): Number of input channels.
		reduction_factor (int): Reduction factor for the 
		bottleneck layer.
		Default is 16.
		dilation (int): Dilation for the 3 X 3 convolutions.
		Default is 4.
	"""
	def __init__(
		self, 
		in_dim: int,
		reduction_factor: int = 16,
		dilation: int = 4,
		) -> None:
		super().__init__()

		bottleneck_dim = in_dim//reduction_factor

		self.reduce_1 = nn.Sequential(
			nn.Conv2d(
				in_channels=in_dim,
				out_channels=bottleneck_dim,
				kernel_size=1,
				),
			nn.BatchNorm2d(bottleneck_dim),
			nn.ReLU(),
			)
		self.convs = nn.Sequential(
			*(2*[
			nn.Conv2d(
				in_channels=bottleneck_dim,
				out_channels=bottleneck_dim,
				kernel_size=3,
				padding=dilation,
				dilation=dilation,
				),
			nn.BatchNorm2d(bottleneck_dim),
			nn.ReLU()]),
			)
		self.reduce_2 = nn.Conv2d(
			in_channels=bottleneck_dim,
			out_channels=1,
			kernel_size=1,
			)
	
	def forward(self, input: torch.Tensor) -> torch.Tensor:
		attention = self.reduce_1(input)
		attention = self.convs(attention)
		attention = self.reduce_2(attention)
		return attention


class BAM(nn.Module):
	"""
	Bottleneck attention module.

	Args:
		in_dim (int): Number of input channels.
		reduction_factor (int): Reduction factor for the bottleneck
		layers of the channel and spatial attention modules.
		Default is 16.
		dilation (int): Dilation for the 3 x 3 convolutions of the spatial
		attention module.
		Default is 4.
	"""
	def __init__(
		self, 
		in_dim: int,
		reduction_factor: int = 16,
		dilation: int = 4,
		) -> None:
		super().__init__()

		self.channel_attention = BAMChannelAttention(
			in_dim=in_dim,
			reduction_factor=reduction_factor,
			)
		self.spatial_attention = BAMSpatialAttention(
			in_dim=in_dim,
			reduction_factor=reduction_factor,
			dilation=dilation,
			)

	def forward(self, input: torch.Tensor) -> torch.Tensor:
		channel_attention = self.channel_attention(input)
		spatial_attention = self.spatial_attention(input)

		attention = channel_attention*spatial_attention
		attention = F.sigmoid(attention)
		attention = attention+1

		output = attention*input
		return output
