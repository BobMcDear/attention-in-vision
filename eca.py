"""
Efficient channel attention (ECA).
"""


from math import log
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


def get_kernel_size(
	in_dim: int,
	beta: int = 1,
	gamma: int = 2,
	) -> int:
	"""
	Calculates the optimal kernel size for ECA.

	Args:
		in_dim (int): Number of input channels.
		beta (int): Beta parameter.
		Default is 1.
		gamma (int): Gamma parameter.
		Default is 2.
	
	Returns (int): Optimal kernel size for ECA given in_dim.
	"""
	t = int((log(in_dim, 2) + beta) / gamma)
	kernel_size = t if t%2 == 1 else t+1
	return kernel_size


class ECA(nn.Module):
	"""
	Efficient channel attention.

	Args:
		beta (int): Beta parameter for calculating the kernel size.
		Default is 1.
		gamma (int): Gamma parameter for calculating the kernel size.
		Default is 2.
		in_dim (Optional[int]): Number of input channels. This value cannot 
		be None if kernel_size is None.
		Default is None.
		kernel_size (Optional[int]): Kernel size. If None, beta and gamma
		are used to calculate the kernel size. Otherwise, beta and gamma
		are ignored, and this value is used as the kernel size instead.
		Default is None.
	"""
	def __init__(
		self, 
		beta: int = 1,
		gamma: int = 2,
		in_dim: Optional[int] = None,
		kernel_size: Optional[int] = None,
		) -> None:
		super().__init__()

		if kernel_size is None:
			kernel_size = get_kernel_size(in_dim,beta, gamma)
		
		self.conv = nn.Conv1d(
			in_channels=1,
			out_channels=1,
			kernel_size=kernel_size,
			padding=kernel_size//2,
			bias=False,
			)

	def forward(self, input: torch.Tensor) -> torch.Tensor:
		avg_pooled = F.adaptive_avg_pool2d(input, 1)
		avg_pooled = avg_pooled.squeeze(2)
		avg_pooled = avg_pooled.transpose(1, 2)

		attention = self.conv(avg_pooled)
		attention = F.sigmoid(attention)
		attention = attention.transpose(1, 2)
		attention = attention.unsqueeze(2)
		
		output = attention*input
		return output
