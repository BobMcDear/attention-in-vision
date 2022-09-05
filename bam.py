from torch import Tensor
from torch.nn import (
	AdaptiveAvgPool2d,
	BatchNorm2d,
	Conv2d,
	Module,
	ReLU,
	Sequential,
	Sigmoid,
	)


class BAMChannelAttention(Module):
	"""
	BAM's channel attention
	"""
	def __init__(
		self, 
		in_dim: int,
		reduction_factor: int = 16,
		) -> None:
		"""
		Sets up the modules
		
		Args:
			in_dim (int): Number of input channels
			reduction_factor (int): Reduction factor for the 
			bottleneck layer.
			Default is 16
		"""
		super().__init__()

		self.avg_pool = AdaptiveAvgPool2d(
			output_size=1,
			)

		bottleneck_dim = in_dim//reduction_factor
		self.mlp = Sequential(
			Conv2d(
				in_channels=in_dim,
				out_channels=bottleneck_dim,
				kernel_size=1,
				),
			BatchNorm2d(
				num_features=bottleneck_dim,
				),
			ReLU(),
			Conv2d(
				in_channels=bottleneck_dim,
				out_channels=in_dim,
				kernel_size=1,
				),
			)
	
	def forward(
		self, 
		input: Tensor,
		) -> Tensor:
		"""
		Runs the input through the module
		
		Args:
			input (Tensor): Input
		
		Returns (Tensor): Result of the module
		"""
		avg_pooled = self.avg_pool(input)
		attention = self.mlp(avg_pooled)
		return attention


class BAMSpatialAttention(Module):
	"""
	BAM's spatial attention
	"""
	def __init__(
		self, 
		in_dim: int,
		reduction_factor: int = 16,
		dilation: int = 4,
		) -> None:
		"""
		Sets up the modules

		Args:
			in_dim (int): Number of input channels
			reduction_factor (int): Reduction factor for the 
			bottleneck layer.
			Default is 16
			dilation (int): Dilation for the 3 X 3 convolutions.
			Default is 4
		"""
		super().__init__()

		bottleneck_dim = in_dim//reduction_factor
		self.reduce_1 = Sequential(
			Conv2d(
				in_channels=in_dim,
				out_channels=bottleneck_dim,
				kernel_size=1,
				),
			BatchNorm2d(
				num_features=bottleneck_dim,
				),
			ReLU(),
			)

		self.layers = Sequential(
			Conv2d(
				in_channels=bottleneck_dim,
				out_channels=bottleneck_dim,
				kernel_size=3,
				padding=dilation,
				dilation=dilation,
				),
			BatchNorm2d(
				num_features=bottleneck_dim,
				),
			ReLU(),
			Conv2d(
				in_channels=bottleneck_dim,
				out_channels=bottleneck_dim,
				kernel_size=3,
				padding=dilation,
				dilation=dilation,
				),
			BatchNorm2d(
				num_features=bottleneck_dim,
				),
			ReLU(),
			)
		
		self.reduce_2 = Conv2d(
			in_channels=bottleneck_dim,
			out_channels=1,
			kernel_size=1,
			)
	
	def forward(
		self, 
		input: Tensor,
		) -> Tensor:
		"""
		Runs the input through the module
		
		Args:
			input (Tensor): Input
		
		Returns (Tensor): Result of the module
		"""
		reduced = self.reduce_1(input)
		attention = self.layers(reduced)
		attention = self.reduce_2(attention)
		return attention


class BAM(Module):
	"""
	Bottleneck attention module
	"""
	def __init__(
		self, 
		in_dim: int,
		reduction_factor: int = 16,
		dilation: int = 4,
		) -> None:
		"""
		Sets up the modules

		Args:
			in_dim (int): Number of input channels
			reduction_factor (int): Reduction factor in the channel 
			and spatial attention modules.
			Default is 16
			dilation (int): Dilation in the spatial attention module.
			Default is 4
		"""
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
		self.sigmoid = Sigmoid()

	def forward(
		self, 
		input: Tensor,
		) -> Tensor:
		"""
		Runs the input through the module
		Args:
			input (Tensor): Input
		
		Returns (Tensor): Result of the module
		"""
		channel_attention = self.channel_attention(input)
		spatial_attention = self.spatial_attention(input)

		attention = channel_attention*spatial_attention
		attention = self.sigmoid(attention)
		attention = attention+1

		output = attention*input
		return output
