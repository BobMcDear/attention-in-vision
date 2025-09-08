from functools import partial

import torch

from bam import BAM
from cbam import CBAM
from condconv import CondConv
from dynamic_conv import DynamicConv
from eca import ECA
from ese import eSE
from ge import GENoParams, GEParams, GEParamsPlus
from mhsa import MHSA
from se import SE
from sk import SK
from splat import SplAt


def test():
	in_dim, out_dim, sz = 64, 128, 14
	input = torch.randn(2, in_dim, sz, sz)
	modules = [
		partial(BAM, in_dim=in_dim),
		partial(CBAM, in_dim=in_dim),
		partial(CondConv, in_dim=in_dim, out_dim=out_dim, groups=2),
		partial(DynamicConv, in_dim=in_dim, out_dim=out_dim),
		partial(ECA, in_dim=in_dim),
		partial(eSE, in_dim=in_dim),
		partial(GENoParams, extent=0),
		partial(GEParams, in_dim=in_dim, extent=0, spatial_dim=(sz, sz)),
		partial(GEParamsPlus, in_dim=in_dim, extent=0, spatial_dim=(sz, sz)),
		partial(SE, in_dim=in_dim),
		partial(SK, in_dim=in_dim, out_dim=out_dim),
		partial(SplAt, in_dim=in_dim, out_dim=out_dim),
		partial(SplAt, in_dim=in_dim, out_dim=out_dim, cardinality=2),
		partial(SplAt, in_dim=in_dim, out_dim=out_dim, radix=1, cardinality=2),
	]
	for mod in modules:
		mod()(input)
	
	input_flat = input.flatten(start_dim=-2, end_dim=-1).transpose(-2, -1)
	MHSA(in_dim=in_dim)(input_flat)
    

if __name__ == '__main__':
	test()
