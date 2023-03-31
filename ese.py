"""
Effective squeeze-and-excitation (eSE).
"""


import torch
from torch import nn
from torch.nn import functional as F


class eSE(nn.Module):
	"""
	Effective squeeze-and-excitation.

	Args:
		in_dim (int): Number of input channels.
	"""
	def __init__(
		self, 
		in_dim: int,
		) -> None:
		super().__init__()

		self.excitation = nn.Sequential(
			nn.Conv2d(
				in_channels=in_dim,
				out_channels=in_dim,
				kernel_size=1,
				),
			nn.Sigmoid(),
			) 

	def forward(self, input: torch.Tensor) -> torch.Tensor:
		squeezed = F.adaptive_avg_pool2d(input, 1)
		attention = self.excitation(squeezed)
		
		output = attention*input
		return output
