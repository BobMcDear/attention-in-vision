"""
Selective kernel unit (SK unit).
"""


import torch
from torch import nn
from torch.nn import functional as F


class Branch(nn.Module):
	"""
	A branch for SK's split module.

	Args:
		in_dim (int): Number of input channels.
		out_dim (int): Number of output channels.
		kernel_size (int): Kernel size. The actual kernel size
		is fixed at 3, but the dilation value is increased to simulate
		larger kernel sizes.
		Default is 3.
		stride (int): Stride.
		Default is 1.
		groups (int): Number of groups.
		Default is 32.
	"""
	def __init__(
		self,
		in_dim: int,
		out_dim: int,
		kernel_size: int = 3,
		stride: int = 1,
		groups: int = 32,
		) -> None:
		super().__init__()

		self.conv = nn.Conv2d(
			in_channels=in_dim,
			out_channels=out_dim,
			kernel_size=3,
			stride=stride,
			padding=kernel_size//2,
			dilation=(kernel_size-1)//2,
			groups=groups,
			bias=False,
			)
		self.bn = nn.BatchNorm2d(out_dim)
		self.relu = nn.ReLU()

	def forward(self, input: torch.Tensor) -> torch.Tensor:
		output = self.conv(input)
		output = self.bn(output)
		output = self.relu(output)
		return output


class Split(nn.Module):
	"""
	SK's split module.

	Args:
		in_dim (int): Number of input channels.
		out_dim (int): Number of output channels.
		n_branches (int): Number of branches.
		Default is 2.
		stride (int): Stride for each branch.
		Default is 1.
		groups (int): Number of groups for each branch.
		Default is 32.
	"""
	def __init__(
		self,
		in_dim: int,
		out_dim: int,
		n_branches: int = 2,
		stride: int = 1,
		groups: int = 32,
		) -> None:
		super().__init__()

		branches = []
		for i in range(1, n_branches+1):
			branch = Branch(
				in_dim=in_dim,
				out_dim=out_dim,
				kernel_size=2*i + 1,
				stride=stride,
				groups=groups,
				)
			branches.append(branch)
		self.branches = nn.ModuleList(branches)

	def forward(self, input: torch.Tensor) -> torch.Tensor:
		outputs = []
		for branch in self.branches:
			output = branch(input)
			outputs.append(output)
		output = torch.stack(outputs, dim=1)
		return output


class Fuse(nn.Module):
	"""
	SK's fuse module.

	Args:
		in_dim (int): Number of channels in each branch.
		reduction_factor (int): Reduction factor for the 
		fully-connected layer.
		Default is 16.
	"""
	def __init__(
		self,
		in_dim: int,
		reduction_factor: int = 16
		) -> None:
		super().__init__()
		
		reduced_dim = in_dim//reduction_factor
		self.fc = nn.Sequential(
			nn.Conv2d(
				in_channels=in_dim,
				out_channels=reduced_dim,
				kernel_size=1,
				bias=False,
				),
			nn.BatchNorm2d(reduced_dim),
			nn.ReLU(),
			)

	def forward(self, input: torch.Tensor) -> torch.Tensor:
		summed = input.sum(1)
		avg_pooled = F.adaptive_avg_pool2d(summed, 1)
		output = self.fc(avg_pooled)
		return output
	

class Select(nn.Module):
	"""
	SK's select module.

	Args:
		in_dim (int): Number of channels in each branch.
		n_branches (int): Number of branches.
		Default is 2.
		reduction_factor (int): Reduction factor for the 
		fully-connected layer of the fuse module.
		Default is 16.
	"""
	def __init__(
		self,
		in_dim: int,
		n_branches: int = 2,
		reduction_factor: int = 16
		) -> None:
		super().__init__()

		self.fc = nn.Conv2d(
			in_channels=in_dim//reduction_factor,
			out_channels=n_branches*in_dim,
			kernel_size=1,
			bias=False,
			)
		self.softmax = nn.Softmax(dim=1)

	def forward(self, input: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
		batch_size, n_branches, in_dim, height, width = input.shape

		attention = self.fc(z)
		attention = attention.reshape(batch_size, n_branches, in_dim, 1, 1)
		attention = self.softmax(attention)

		output = attention*input
		output = output.sum(1)
		return output


class SK(nn.Module):
	"""
	Selective kernel module.

	Args:
		in_dim (int): Number of input channels.
		out_dim (int): Number of output channels.
		n_branches (int): Number of branches.
		Default is 2.
		stride (int): Stride for each branch.
		Default is 1.
		groups (int): Number of groups for each branch.
		Default is 32.
		reduction_factor (int): Reduction factor for the 
		fully-connected layer for the fuse module.
		Default is 16.
	"""
	def __init__(
		self,
		in_dim: int,
		out_dim: int,
		n_branches: int = 2,
		stride: int = 1,
		groups: int = 32,
		reduction_factor: int = 16
		) -> None:
		super().__init__()

		self.split = Split(
			in_dim=in_dim,
			out_dim=out_dim,
			n_branches=n_branches,
			stride=stride,
			groups=groups,
			)
		self.fuse = Fuse(
			in_dim=out_dim,
			reduction_factor=reduction_factor,
			)
		self.select = Select(
			in_dim=out_dim,
			n_branches=n_branches,
			reduction_factor=reduction_factor,
			)
	
	def forward(self, input: torch.Tensor) -> torch.Tensor:
		branches = self.split(input)
		z = self.fuse(branches)
		output = self.select(branches, z)
		return output
