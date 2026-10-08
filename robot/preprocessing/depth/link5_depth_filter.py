"""Observed-only depth support; no pair cropping, CAD fitting or depth replacement.

Boundary depth halos are often smooth in image space, so isolated-spike checks
miss them. Test their actual 3D sampling density instead, accounting for the
camera pixel footprint at each sample's depth. A conservative robust threshold
rejects sparsely supported tails without imposing a normal-shape envelope.
"""
from dataclasses import asdict, dataclass

import cv2
import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class Config:
    erosion_pixels: int = 2  # original source-image pixels, before grid alignment
    neighbors: int = 32
    density_mad_multiplier: float = 6.0
    density_median_multiplier: float = 3.0
    minimum_pixels: int = 256
    minimum_depth_retained_fraction: float = .5

    def validate(self):
        if self.erosion_pixels not in (0,1,2):raise ValueError('source mask erosion must be 0, 1 or 2 pixels')
        if self.neighbors<4 or self.minimum_pixels<128:raise ValueError('insufficient neighborhood or cloud support')
        if self.density_mad_multiplier<=0 or self.density_median_multiplier<=1:raise ValueError('invalid density threshold')
        if not 0<self.minimum_depth_retained_fraction<=1:raise ValueError('invalid retained-fraction gate')
        return self


def settings(config):
    return asdict(config.validate())


def xyz_from_depth(depth, support, k):
    yy,xx=np.where(support);z=np.asarray(depth)[yy,xx]
    xyz=np.column_stack(((xx-k[0,2])*z/k[0,0],(yy-k[1,2])*z/k[1,1],z)).astype(np.float32)
    return xyz,np.column_stack((yy,xx))


def filter_depth(depth, source_mask, k, config=None):
    config=(config or Config()).validate()
    depth=np.asarray(depth);source_mask=np.asarray(source_mask,bool);k=np.asarray(k)
    if depth.ndim!=2 or source_mask.ndim!=2 or k.shape!=(3,3):raise ValueError('invalid observation grids/intrinsics')
    if not np.isfinite(k).all() or k[0,0]<=0 or k[1,1]<=0:raise ValueError('invalid focal lengths')
    h,w=depth.shape
    align=lambda m: m.copy() if m.shape==depth.shape else cv2.resize(m.astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST).astype(bool)
    grid=align(source_mask)
    if config.erosion_pixels:
        radius=config.erosion_pixels
        eroded_source=cv2.erode(source_mask.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1)),
            borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
    else:eroded_source=source_mask.copy()
    eroded=align(eroded_source)&grid
    finite=np.isfinite(depth)&(depth>0);candidates=eroded&finite
    xyz,pixels=xyz_from_depth(depth,candidates,k)
    score=np.full(depth.shape,np.nan,np.float32);outliers=np.zeros(depth.shape,bool)
    median=mad=threshold=None
    if len(xyz)>config.neighbors:
        distances,_=cKDTree(xyz).query(xyz,k=config.neighbors+1,workers=1)
        # Distances are used only to choose observed pixels; XYZ is never moved.
        pixel_footprint=xyz[:,2]/np.sqrt(float(k[0,0])*float(k[1,1]))
        density=distances[:,1:].mean(1)/pixel_footprint
        median=float(np.median(density));mad=float(1.4826*np.median(np.abs(density-median)))
        threshold=max(config.density_median_multiplier*median,median+config.density_mad_multiplier*mad)
        score[pixels[:,0],pixels[:,1]]=density.astype(np.float32)
        bad=density>threshold;outliers[pixels[bad,0],pixels[bad,1]]=True
    valid=candidates&~outliers;rejected=grid&~valid
    reasons=np.zeros(depth.shape,np.uint8)
    reasons[grid&~finite]=1;reasons[grid&~eroded&finite]=2;reasons[outliers]=3
    info=dict(method='ray_normalized_3d_knn_density_v1',**settings(config),
        input_mask_pixel_count=int(grid.sum()),finite_positive_depth_count=int((grid&finite).sum()),
        invalid_depth_pixel_count=int((grid&~finite).sum()),erosion_rejected_pixel_count=int((grid&~eroded&finite).sum()),
        density_rejected_pixel_count=int(outliers.sum()),rejected_depth_pixel_count=int(rejected.sum()),
        final_valid_pixel_count=int(valid.sum()),retained_fraction=float(valid.sum()/max(1,grid.sum())),
        density_median=median,density_scaled_mad=mad,density_threshold=threshold,
        threshold_rule=f'max({config.density_median_multiplier} * median, median + {config.density_mad_multiplier} * 1.4826 * MAD) of mean{config.neighbors}-neighbor distance / (depth / sqrt(fx*fy))',
        density_evaluated=threshold is not None,depth_smoothing=False,depth_replacement=False,
        simple3d_cropping=False,cad_distance_filter=False,global_depth_percentile_clipping=False,
        erosion_coordinate_grid='original source image; constant-zero border')
    return dict(mask=grid,eroded_mask=eroded,valid=valid,rejected=rejected,rejection_reason=reasons,density_score=score,info=info)
