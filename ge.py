from math import log2
from typing import (
	Optional,
	Tuple,
	Union,
	)

from torch import Tensor
from torch.nn import (
	AdaptiveAvgPool2d,
	AvgPool2d,
	BatchNorm2d,
	Conv2d,
	Module,
	ReLU,
	Sequential,
	Sigmoid,
	)
from torch.nn.functional import interpolate


class GENoParams(Module):
	"""
	Gather-excite with no parameters
	"""
	def __init__(
		self,
		extent: int,
		) -> None:
		"""
		Sets up the modules

		Args:
			extent (int): Extent. 0 for a global 
			extent
		"""
		super().__init__()

		if extent_ratio == 0:
			self.gather = AdaptiveAvgPool2d(
				output_size=1,
				)

		else:
			kernel_size = 2*extent - 1
			padding = kernel_size//2
			self.gather = AvgPool2d(
				kernel_size=kernel_size,
				stride=extent,
				padding=padding,
				count_include_pad=False,
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
		batch_size, in_dim, height, width = input.shape

		gathered = self.gather(input)

		attention = interpolate(
			input=gathered,
			size=(height, width),
			mode='nearest',
			)
		attention = self.sigmoid(attention)

		output = attention*input
		return output


class GEParams(Module):
	"""
	Gather-excite with parameters
	"""
	def __init__(
		self,
		in_dim: int,
		extent: int,
		spatial_dim: Optional[Union[Tuple[int, int], int]] = None,
		) -> None:
		"""
		Sets up the modules

		Args:
			in_dim (int): Number of input channels
			extent (int): Extent. 0 for a global
			extent
			spatial_dim (Optional[Union[Tuple[int, int], int]]):
			Spatial dimension of the input, required for a global 
			extent.
			Default is None
		"""
		super().__init__()

		if extent_ratio == 0:
			self.gather = Sequential(
				Conv2d(
					in_channels=in_dim,
					out_channels=in_dim,
					kernel_size=spatial_dim,
					groups=in_dim,
					bias=False,
					),
				BatchNorm2d(
					num_features=in_dim,
					),
				)

		else:
			n_layers = int(log2(extent))
			layers = n_layers * [
				Conv2d(
					in_channels=in_dim,
					out_channels=in_dim,
					kernel_size=3,
					stride=2,
					padding=1,
					groups=in_dim,
					bias=False,
					),
				BatchNorm2d(
					num_features=in_dim,
					),
				ReLU(),
				]
			layers = layers[::-1]
			self.gather = Sequential(*layers)

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
		batch_size, in_dim, height, width = input.shape

		gathered = self.gather(input)

		attention = interpolate(
			input=gathered,
			size=(height, width),
			mode='nearest',
			)
		attention = self.sigmoid(attention)

		output = attention*input
		return output


class GEParamsPlus(Module):
	"""
	Gather-excite with parameters, including for the excite unit
	"""
	def __init__(
		self,
		in_dim: int,
		extent: int,
		spatial_dim: Optional[Union[Tuple[int, int], int]] = None,
		) -> None:
		"""
		Sets up the modules

		Args:
			in_dim (int): Number of input channels
			extent (int): Extent. 0 for a global
			extent
			spatial_dim (Optional[Union[Tuple[int, int], int]]):
			Spatial dimension of the input, required for a global 
			extent.
			Default is None
		"""
		super().__init__()

		if extent == 0:
			self.gather = Sequential(
				Conv2d(
					in_channels=in_dim,
					out_channels=in_dim,
					kernel_size=spatial_dim,
					groups=in_dim,
					bias=False,
					),
				BatchNorm2d(
					num_features=in_dim,
					),
				)

		else:
			n_layers = int(log2(extent))
			layers = n_layers * [
				Conv2d(
					in_channels=in_dim,
					out_channels=in_dim,
					kernel_size=3,
					stride=2,
					padding=1,
					groups=in_dim,
					bias=False,
					),
				BatchNorm2d(
					num_features=in_dim,
					),
				ReLU(),
				]
			layers = layers[::-1]
			self.gather = Sequential(*layers)
		
		bottleneck_dim = in_dim//16
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
		batch_size, in_dim, height, width = input.shape

		gathered = self.gather(input)

		attention = self.mlp(gathered)
		attention = interpolate(
			input=attention,
			size=(height, width),
			mode='nearest',
			)
		attention = self.sigmoid(attention)

		output = attention*input
		return output
