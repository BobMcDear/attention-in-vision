"""
Conditionally parameterized convolution (ConvConv).
"""


import torch
from torch import nn
from torch.nn import functional as F


class Router(nn.Module):
	"""
	Routing function for CondConv.

	Args:
		in_dim (int): Number of input channels.
		n_experts (int): Number of experts.
		Default is 8.
	"""
	def __init__(
		self,
		in_dim: int,
		n_experts: int = 8,
		) -> None:
		super().__init__()

		self.linear = nn.Conv2d(
			in_channels=in_dim,
			out_channels=n_experts,
			kernel_size=1,
			)

	def forward(self, input: torch.Tensor) -> torch.Tensor:
		avg_pooled = F.adaptive_avg_pool2d(input, 1)
		attention = self.linear(avg_pooled)
		attention = F.sigmoid(attention)
		attention = attention.flatten(1, 3)
		return attention


class Combine(nn.Module):
	"""
	Combines multiple convolutional layers given attention values.

	Args:
		in_dim (int): Number of input channels.
		out_dim (int): Number of output channels.
		kernel_size (int): Kernel size of each convolution.
		Default is 3
		stride (int): Stride of each convolution.
		Default is 1
		padding (int): Padding of each convolution.
		Default is 1
		dilation (int): Dilation of each convolution.
		Default value is 1.
		groups (int): Number of groups of each convolution.
		Default is 1
		bias (bool): Whether each convolution should have a bias term.
		Default is True
		n_experts (int): Number of experts.
		Default is 8
	"""
	def __init__(
		self,
		in_dim: int,
		out_dim: int,
		kernel_size: int = 3,
		stride: int = 1,
		padding: int = 1,
		dilation: int = 1,
		groups: int = 1,
		bias: bool = True,
		n_experts: int = 8,
		) -> None:
		super().__init__()

		self.out_dim = out_dim
		self.stride = stride
		self.padding = padding
		self.dilation = dilation
		self.groups = groups

		weights = torch.randn(
			n_experts,
			out_dim,
			in_dim//groups,
			kernel_size,
			kernel_size,
			)
		self.weights = nn.Parameter(weights)
		self.bias = nn.Parameter(torch.randn(n_experts, out_dim)) if bias else None
		
	def forward(self, input: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
		batch_size, in_dim, height, width = input.shape
		input = input.reshape(1, batch_size*in_dim, height, width)

		weights = torch.einsum(
			'bn,noihw->boihw',
			attention,
			self.weights,
			)
		weights = weights.reshape(batch_size*self.out_dim, *weights.shape[2:])

		bias = None
		if self.bias is not None:
			bias = torch.einsum('bn,no->bo', attention, self.bias)
			bias = bias.reshape(batch_size*self.out_dim)

		output = F.conv2d(
			input=input,
			weight=weights,
			bias=bias,
			stride=self.stride,
			padding=self.padding,
			dilation=self.dilation,
			groups=batch_size*self.groups,
			)
		output = output.reshape(batch_size, self.out_dim, *output.shape[2:])
		return output


class CondConv(nn.Module):
	"""
	Conditionally parameterized convolution.

	Args:
		in_dim (int): Number of input channels.
		out_dim (int): Number of output channels.
		kernel_size (int): Kernel size of each convolution.
		Default is 3.
		stride (int): Stride of each convolution.
		Default is 1.
		padding (int): Padding of each convolution.
		Default is 1.
		dilation (int): Dilation of each convolution.
		Default value is 1.
		groups (int): Number of groups of each convolution.
		Default is 1.
		bias (bool): Whether each convolution should have a bias term.
		Default is True.
		n_experts (int): Number of experts.
		Default is 8.
	"""
	def __init__(
		self,
		in_dim: int,
		out_dim: int,
		kernel_size: int = 3,
		stride: int = 1,
		padding: int = 1,
		dilation: int = 1,
		groups: int = 1,
		bias: bool = True,
		n_experts: int = 8,
		) -> None:
		super().__init__()

		self.router = Router(
			in_dim=in_dim,
			n_experts=n_experts,
			)
		self.combine = Combine(
			in_dim=in_dim,
			out_dim=out_dim,
			kernel_size=kernel_size,
			stride=stride,
			padding=padding,
			dilation=dilation,
			groups=groups,
			bias=bias,
			n_experts=n_experts,
			)

	def forward(self, input: torch.Tensor) -> torch.Tensor:
		attention = self.router(input)
		output = self.combine(input, attention)
		return output
