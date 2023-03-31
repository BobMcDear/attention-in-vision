"""
Multi-headed self-attention (MHSA).
"""


from typing import Tuple
from math import sqrt

import torch
from torch import nn
from torch.nn import functional as F


class QKV(nn.Module):
	"""
	Extracts queries, keys, and values for MHSA.

	Args:
		in_dim (int): Dimension of input.
		n_heads (int): Number of heads.
		Default is 8.
	"""
	def __init__(
		self,
		in_dim: int,
		n_heads: int = 8,
		) -> None:
		super().__init__()

		self.n_heads = n_heads
		self.qkv_dim = in_dim//n_heads
		self.to_qkv = nn.Linear(
			in_features=in_dim,
			out_features=3*in_dim,
			)
		
	def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, ...]:
		batch_size, n_tokens, in_dim = input.shape
		qkv = self.to_qkv(input)
		qkv = qkv.reshape(batch_size, n_tokens, 3, self.n_heads, self.qkv_dim)
		qkv = qkv.permute(2, 0, 3, 1, 4)
		return qkv.unbind(dim=0)


def get_attention(queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
	"""
	Calculates scaled dot-product attention given queries and keys.

	Args:
		queries (torch.Tensor): Queries.
		keys (torch.Tensor): Keys.

	Returns (torch.Tensor): Attention calculated using the provided queries
	and keys.
	"""
	attention = (queries @ keys.transpose(-2, -1)) / sqrt(queries.shape[-1])
	attention = F.softmax(attention, dim=-1)
	return attention


class MHSA(nn.Module):
	"""
	Multi-headed self-attention.

	Args:
		in_dim (int): Dimension of input
		n_heads (int): Number of heads.
		Default is 8
	"""
	def __init__(
		self,
		in_dim: int,
		n_heads: int = 8,
		) -> None:
		super().__init__()

		self.n_heads = n_heads
		self.to_qkv = QKV(
			in_dim=in_dim,
			n_heads=n_heads,
			)
		self.to_output = nn.Linear(
			in_features=in_dim,
			out_features=in_dim,
			)

	def forward(self, input: torch.Tensor) -> torch.Tensor:
		batch_size, n_tokens, in_dim = input.shape

		queries, keys, values = self.to_qkv(input)
		attention = get_attention(
			queries=queries,
			keys=keys,
			)
		output = attention @ values

		output = output.transpose(1, 2)
		output = output.reshape(batch_size, n_tokens, in_dim)
		output = self.to_output(output)

		return output
