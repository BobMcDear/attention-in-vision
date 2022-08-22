from torch import Tensor
from torch.nn import (
	AdaptiveAvgPool2d,
	Conv1d,
	Module,
	Sequential,
	Sigmoid,
	)


class ECA(Module):
	"""
	Efficient channel attention
	"""
	def __init__(
		self, 
		kernel_size: int = 3,
		) -> None:
		"""
		Sets up the modules

		Args:
			kernel_size (int): Kernel size.
			Default is 3
		"""
		super().__init__()

		padding = kernel_size//2
		self.avg_pool = AdaptiveAvgPool2d(
			output_size=1,
			)
		self.conv = Conv1d(
			in_channels=1,
			out_channels=1,
			kernel_size=kernel_size,
			padding=padding,
			bias=False,
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

		avg_pooled = avg_pooled.squeeze(2)
		avg_pooled = avg_pooled.transpose(1, 2)

		attention = self.conv(avg_pooled)
		attention = self.sigmoid(attention)

		attention = attention.transpose(1, 2)
		attention = attention.unsqueeze(2)
		
		output = attention*input
		return output
