"""Frame-zero pair selector interface; indices refer to supplied eligible points."""
import numpy as np
from .balanced_v0 import select_balanced
from .refine_v1 import choose_pairs, mask_geometry, jsonable
METHODS=('balanced_v0','refine_v1')

def select_pairs(method, xy, xyz, mask0):
    xy=np.asarray(xy,float)
    if method=='balanced_v0':
        return select_balanced(xy,xyz)
    if method=='refine_v1':
        geometry=mask_geometry(np.asarray(mask0,bool))
        pairs,stats,_,_=choose_pairs(xy,geometry)
        stats.update(geometry_flags=geometry['flags'],transition=geometry['transition'])
        return jsonable(pairs),jsonable(stats)
    raise ValueError(f'Unknown method: {method}')
