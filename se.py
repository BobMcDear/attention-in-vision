"""
Squeeze-and-excitation (SE).
"""


import torch
from torch import nn
from torch.nn import functional as F


class SE(nn.Module):
	"""
	Squeeze-and-excitation.

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
		self.excitation = nn.Sequential(
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
			nn.Sigmoid(),
			) 

	def forward(self, input: torch.Tensor) -> torch.Tensor:
		squeezed = F.adaptive_avg_pool2d(input, 1)
		attention = self.excitation(squeezed)
		output = attention*input
		return output
