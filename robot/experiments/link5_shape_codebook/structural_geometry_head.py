"""Geometry-conditioned restoration head for the Link5 research experiment.

Keep the strict frozen FCGF and original codebook network. The extra learned head
also sees fixed-reference XYZ, nearest observed normal geometry and a whole-link
profile. This addresses the lost length/curvature context in local descriptors.
Only the four normal clouds populate its geometry bank. Training excludes the
current source from auxiliary nearest-neighbor queries to avoid exact-self-match
shortcuts. This is an explicit architecture adaptation, not a paper reproduction.
"""
import numpy as np
from scipy.spatial import cKDTree
import torch
from torch import nn


class StructuralGeometryDetector(nn.Module):
    PROFILE_BINS = 12

    def __init__(self, native, normal_points, geometry):
        super().__init__(); self.native = native
        if len(normal_points) != 4:raise ValueError('exact four normal references required')
        for i, points in enumerate(normal_points):
            self.register_buffer(f'normal_{i}', points.detach().clone())
        u = normal_points[0].new_tensor(geometry['axis'])
        basis = torch.eye(3, device=u.device, dtype=u.dtype)[u.abs().argmin()]
        v = torch.linalg.cross(u, basis); v = v / v.norm(); w = torch.linalg.cross(u, v)
        self.register_buffer('frame_basis', torch.stack([u, v, w], dim=1))
        self.register_buffer('frame_anchor', u.new_tensor(geometry['anchor']))
        self.register_buffer('frame_length', u.new_tensor(geometry['length']))
        # Twelve bins: two cross-section center coordinates, radius and support.
        # Global quantiles/centroid are reported as features, never normalization.
        self.context_size = self.PROFILE_BINS * 4 + 9
        input_size = native.feature_dim + 3 + 4 + self.context_size + 4
        self.geometry_mlp = nn.Sequential(nn.Linear(input_size, 128), nn.GELU(),
                                          nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 64), nn.GELU())
        self.offset_head = nn.Linear(64, 3); self.mask_head = nn.Linear(64, 1)
        self.reference_exclusion = None
        self._trees = {}; self._banks = {}; self._point_features = None
        self.native.encoder.register_forward_hook(self._capture_features)

    @property
    def codebook(self):return self.native.codebook

    @property
    def encoder(self):return self.native.encoder

    def update_codebook(self, points):return self.native.update_codebook(points)

    def _capture_features(self, module, inputs, result):self._point_features = result

    def _reference(self, exclusion):
        if exclusion not in (None, 0, 1, 2, 3):raise ValueError('invalid normal source exclusion')
        if exclusion not in self._trees:
            arrays = [getattr(self, f'normal_{i}').detach().cpu().numpy() for i in range(4) if i != exclusion]
            bank = np.concatenate(arrays)
            self._banks[exclusion] = bank
            self._trees[exclusion] = cKDTree(bank)
        return self._trees[exclusion], self._banks[exclusion]

    @torch.no_grad()
    def geometry_features(self, points):
        coordinates = (points - self.frame_anchor) @ self.frame_basis
        tree, bank = self._reference(self.reference_exclusion)
        outputs = []
        for xyz, local in zip(points, coordinates):
            _, nearest = tree.query(xyz.detach().cpu().numpy(), k=1, workers=1)
            difference = torch.as_tensor(bank[nearest], device=xyz.device, dtype=xyz.dtype) - xyz
            neighbor = torch.cat([difference, difference.norm(dim=-1, keepdim=True)], dim=-1)
            # Fixed normal axis/length; absolute quantiles retain shortening.
            global_values = torch.cat([torch.quantile(local, local.new_tensor([.01,.99]), dim=0).reshape(-1), local.mean(dim=0)])
            profile = []
            for b in range(self.PROFILE_BINS):
                low = self.frame_length * b / self.PROFILE_BINS
                high = self.frame_length * (b+1) / self.PROFILE_BINS
                membership = (local[:,0] >= low) & (local[:,0] < high)
                sample = local[membership,1:]
                if len(sample):
                    center = sample.median(dim=0).values
                    radius = (sample-center).norm(dim=-1).median()
                    profile.append(torch.cat([center, radius[None], membership.float().mean()[None]]))
                else:profile.append(local.new_zeros(4))
            context = torch.cat([global_values, *profile])
            outputs.append(torch.cat([local, neighbor, context[None].expand(len(local), -1)], dim=-1))
        return torch.stack(outputs)

    def forward(self, points):
        offset, logits, aux = self.native(points)
        features = self._point_features
        if features is None or features.shape[:2] != points.shape[:2]:raise ValueError('strict pretrained point features missing')
        shape = self.geometry_features(points)
        learned = self.geometry_mlp(torch.cat([features, shape, offset, logits[...,None]], dim=-1))
        # Predict a new correction/probability using both native and shape context.
        # The raw score formula is unchanged; no hand-written anomaly scores.
        return self.offset_head(learned), self.mask_head(learned).squeeze(-1), aux


def wrap_geometry_model(native, samples, geometry):
    return StructuralGeometryDetector(native, [s[1] for s in samples], geometry).to(samples[0][1].device)
