"""
Gather-excite (GE).
"""


from math import log2
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F


class GENoParams(nn.Module):
	"""
	Gather-excite with no parameters.

	Args:
		extent (int): Extent. 0 for a global 
		extent.
	"""
	def __init__(
		self,
		extent: int,
		) -> None:
		super().__init__()

		if extent == 0:
			self.gather = nn.AdaptiveAvgPool2d(1)

		else:
			kernel_size = 2*extent - 1
			self.gather = nn.AvgPool2d(
				kernel_size=kernel_size,
				stride=extent,
				padding=kernel_size//2,
				count_include_pad=False,
				)

	def forward(self, input: torch.Tensor) -> torch.Tensor:
		gathered = self.gather(input)
		attention = F.interpolate(
			input=gathered,
			size=input.shape[-2:],
			mode='nearest',
			)
		attention = F.sigmoid(attention)

		output = attention*input
		return output


class GEParams(nn.Module):
	"""
	Gather-excite with parameters.
	
	Args:
		in_dim (int): Number of input channels.
		extent (int): Extent. 0 for a global
		extent.
		spatial_dim (Optional[Union[Tuple[int, int], int]]):
		Spatial dimension of the input, required for a global 
		extent.
		Default is None.
	"""
	def __init__(
		self,
		in_dim: int,
		extent: int,
		spatial_dim: Optional[Union[Tuple[int, int], int]] = None,
		) -> None:
		super().__init__()

		if extent == 0:
			self.gather = nn.Sequential(
				nn.Conv2d(
					in_channels=in_dim,
					out_channels=in_dim,
					kernel_size=spatial_dim,
					groups=in_dim,
					bias=False,
					),
				nn.BatchNorm2d(in_dim),
				)

		else:
			n_layers = int(log2(extent))
			layers = n_layers * [
				nn.Conv2d(
					in_channels=in_dim,
					out_channels=in_dim,
					kernel_size=3,
					stride=2,
					padding=1,
					groups=in_dim,
					bias=False,
					),
				nn.BatchNorm2d(in_dim),
				nn.ReLU(),
				]
			layers = layers[:-1]
			self.gather = nn.Sequential(*layers)

	def forward(self, input: torch.Tensor) -> torch.Tensor:
		gathered = self.gather(input)
		attention = F.interpolate(
			input=gathered,
			size=input.shape[-2:],
			mode='nearest',
			)
		attention = F.sigmoid(attention)

		output = attention*input
		return output


class GEParamsPlus(nn.Module):
	"""
	Gather-excite with parameters, including for the excite unit.

	Args:
		in_dim (int): Number of input channels.
		extent (int): Extent. 0 for a global
		extent.
		reduction_factor (int): Reduction factor for the 
		bottleneck layer of the excite module.
		Default is 16.
		spatial_dim (Optional[Union[Tuple[int, int], int]]):
		Spatial dimension of the input, required for a global 
		extent.
		Default is None.
	"""
	def __init__(
		self,
		in_dim: int,
		extent: int,
		reduction_factor: int = 16,
		spatial_dim: Optional[Union[Tuple[int, int], int]] = None,
		) -> None:
		super().__init__()

		if extent == 0:
			self.gather = nn.Sequential(
				nn.Conv2d(
					in_channels=in_dim,
					out_channels=in_dim,
					kernel_size=spatial_dim,
					groups=in_dim,
					bias=False,
					),
				nn.BatchNorm2d(in_dim),
				)

		else:
			n_layers = int(log2(extent))
			layers = n_layers * [
				nn.Conv2d(
					in_channels=in_dim,
					out_channels=in_dim,
					kernel_size=3,
					stride=2,
					padding=1,
					groups=in_dim,
					bias=False,
					),
				nn.BatchNorm2d(in_dim),
				nn.ReLU(),
				]
			layers = layers[:-1]
			self.gather = nn.Sequential(*layers)

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
		gathered = self.gather(input)
		attention = self.mlp(gathered)
		attention = F.interpolate(
			input=attention,
			size=input.shape[-2:],
			)
		attention = F.sigmoid(attention)

		output = attention*input
		return output
