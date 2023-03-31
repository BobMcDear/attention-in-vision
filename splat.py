import torch
from torch import nn


class Branches(nn.Module):
	"""
	SplAt's branches.

	Args:
		in_dim (int): Number of input channels.
		out_dim (int): Number of output channels.
		kernel_size (int): Kernel size.
		Default is 3.
		stride (int): Stride.
		Default is 1.
		cardinality (int): Cardinality.
		Default is 1.
		radix (int): Radix.
		Default is 2.
	"""
	def __init__(
		self,
		in_dim: int,
		out_dim: int,
		kernel_size: int = 3,
		stride: int = 1,
		cardinality: int = 1,
		radix: int = 2,
		) -> None:
		super().__init__()

		self.radix = radix

		branches_out_dim = radix*out_dim
		self.branches = nn.Sequential(
			nn.Conv2d(
				in_channels=in_dim,
				out_channels=branches_out_dim,
				kernel_size=kernel_size,
				stride=stride,
				padding=kernel_size//2,
				groups=cardinality*radix,
				bias=False,
				),
			nn.BatchNorm2d(branches_out_dim),
			nn.ReLU(),
			)

	def forward(self, input: torch.Tensor) -> torch.Tensor:
		batch_size, in_dim, height, width = input.shape
		output = self.branches(input)
		output = output.reshape(batch_size, self.radix, -1, height, width)
		return output


class RadixSoftmax(nn.Module):
	"""
	Softmax applied over the radix dimension.

	Args:
		cardinality (int): Cardinality.
		Default is 1.
		radix (int): Radix. If 1, sigmoid is applied.
		Default is 2.
	"""
	def __init__(
		self,
		cardinality: int = 1,
		radix: int = 2,
		) -> None:
		super().__init__()

		self.radix = radix
		self.cardinality = cardinality

		if radix == 1:
			self.gate = nn.Sigmoid()
		
		else:
			self.gate = nn.Softmax(dim=1)

	def forward(self, input: torch.Tensor) -> torch.Tensor:
		batch_size, in_dim, height, width = input.shape

		output = input.reshape(batch_size, self.cardinality, self.radix, -1)
		output = output.transpose(1, 2)
		output = self.gate(output)
		return output


class SplAt(nn.Module):
	"""
	Split attention.

	Args:
		in_dim (int): Number of input channels.
		out_dim (int): Number of output channels.
		kernel_size (int): Kernel size.
		Default is 3.
		stride (int): Stride.
		Default is 1.
		cardinality (int): Cardinality.
		Default is 1.
		radix (int): Radix.
		Default is 2.
		reduction_factor (int): Reduction factor for the 
		fully-connected layer.
		Default is 4.
	"""
	def __init__(
		self,
		in_dim: int,
		out_dim: int,
		kernel_size: int = 3,
		stride: int = 1,
		cardinality: int = 1,
		radix: int = 2,
		reduction_factor: int = 4,
		) -> None:
		super().__init__()

		self.stride = stride
		self.radix = radix

		branches_out_dim = radix*out_dim
		reduced_dim = branches_out_dim//reduction_factor
		self.branches = Branches(
			in_dim=in_dim,
			out_dim=out_dim,
			kernel_size=kernel_size,
			stride=stride,
			cardinality=cardinality,
			radix=radix,
			)
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.mlp = nn.Sequential(
			nn.Conv2d(
				in_channels=out_dim,
				out_channels=reduced_dim,
				kernel_size=1,
				groups=cardinality,
				bias=False,
				),
			nn.BatchNorm2d(reduced_dim),
			nn.ReLU(),
			nn.Conv2d(
				in_channels=reduced_dim,
				out_channels=branches_out_dim,
				kernel_size=1,
				groups=cardinality,
				)
			)
		self.radix_softmax = RadixSoftmax(
			cardinality=cardinality,
			radix=radix,
			)

	def forward(self, input: torch.Tensor) -> torch.Tensor:
		batch_size = len(input)

		branches_output = self.branches(input)
		branches_summed = branches_output.sum(1)
		avg_pooled = self.avg_pool(branches_summed)
		
		attention = self.mlp(avg_pooled)
		attention = self.radix_softmax(attention)
		attention = attention.reshape(batch_size, self.radix, -1, 1, 1)
		
		output = attention*branches_output
		output = output.sum(1)
		return output
