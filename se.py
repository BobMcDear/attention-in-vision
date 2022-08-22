from torch import Tensor
from torch.nn import (
	AdaptiveAvgPool2d,
	Conv2d,
	Module,
	ReLU,
	Sequential,
	Sigmoid,
	)


class SE(Module):
	"""
	Squeeze-and-excitation
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

		bottleneck_dim = in_dim//reduction_factor
		self.squeeze = AdaptiveAvgPool2d(
			output_size=1,
			)
		self.excitation = Sequential(
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
			Sigmoid(),
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
		squeezed = self.squeeze(input)
		attention = self.excitation(squeezed)
		
		output = attention*input
		return output
