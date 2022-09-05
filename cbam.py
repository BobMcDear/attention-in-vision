from torch import (
	Tensor,
	cat,
	)
from torch.nn import (
	AdaptiveAvgPool2d,
	AdaptiveMaxPool2d,
	Conv2d,
	Module,
	ReLU,
	Sequential,
	Sigmoid,
	)


class CBAMChannelAttention(Module):
	"""
	CBAM's channel attention
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
		self.max_pool = AdaptiveMaxPool2d(
			output_size=1,
			)

		bottleneck_dim = in_dim//reduction_factor
		self.mlp = Sequential(
			Conv2d(
				in_channels=in_dim,
				out_channels=bottleneck_dim,
				kernel_size=1,
				),
			ReLU(),
			Conv2d(
				in_channels=bottleneck_dim,
				out_channels=in_dim,
				kernel_size=1,
				),
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
		avg_pooled = self.avg_pool(input)
		max_pooled = self.max_pool(input)

		avg_attention = self.mlp(avg_pooled)
		max_attention = self.mlp(max_pooled)
		
		attention = avg_attention+max_attention
		attention = self.sigmoid(attention)
		
		output = attention*input
		return output


class ChannelAvgPool(Module):
	"""
	Average pool along the channel axis
	"""
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
		output = input.mean(
			dim=1,
			keepdim=True,
			)
		return output


class ChannelMaxPool(Module):
	"""
	Max pool along the channel axis
	"""
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
		output = input.max(
			dim=1,
			keepdim=True,
			).values
		return output


class CBAMSpatialAttention(Module):
	"""
	CBAM's spatial attention
	"""
	def __init__(
		self, 
		kernel_size: int = 7,
		) -> None:
		"""
		Sets up the modules
		Args:
			kernel_size (int): Kernel size.
			Default is 7
		"""
		super().__init__()

		self.avg_pool = ChannelAvgPool()
		self.max_pool = ChannelMaxPool()

		padding = kernel_size//2
		self.conv = Conv2d(
			in_channels=2,
			out_channels=1,
			kernel_size=kernel_size,
			padding=padding,
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
		avg_pooled = self.avg_pool(input)
		max_pooled = self.max_pool(input)
		pooled = cat(
			tensors=[avg_pooled, max_pooled],
			dim=1,
			)

		attention = self.conv(pooled)
		attention = self.sigmoid(attention)		
		
		output = attention*input
		return output


class CBAM(Module):
	"""
	Convolutional block attention module
	"""
	def __init__(
		self, 
		in_dim: int,
		reduction_factor: int = 16,
		kernel_size: int = 7,
		) -> None:
		"""
		Sets up the modules
		Args:
			in_dim (int): Number of input channels
			reduction_factor (int): Reduction factor for the 
			bottleneck layer in the channel attention module.
			Default is 16
			kernel_size (int): Kernel size in the spatial 
			attention module.
			Default is 7
		"""
		super().__init__()

		self.channel_attention = CBAMChannelAttention(
			in_dim=in_dim,
			reduction_factor=reduction_factor,
			)
		self.spatial_attention = CBAMSpatialAttention(
			kernel_size=kernel_size,
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
		output = self.channel_attention(input)
		output = self.spatial_attention(output)
		return output
