from typing import Callable, Dict, List
from enum import Enum, unique, auto

import numpy as np


@unique
class Q(Enum):
    """Sawyer quantities"""
    eef_pos = auto()
    gripper_state = auto()
    handle_pos = auto()
    handle_quat = auto()
    goal = auto()
    unused = auto()


quants_to_sizes = {
    Q.eef_pos: 3,
    Q.gripper_state: 1,
    Q.handle_pos: 3,
    Q.handle_quat: 4,
    Q.unused: 25,
    Q.goal: 3,
}


quants_to_bounds = {q: np.full((quants_to_sizes[q],), 1.0)
                    for q in Q}


OBS_SPECS = {
    'default': [Q.eef_pos, Q.gripper_state, Q.handle_pos, Q.handle_quat, Q.unused, Q.goal],
}


class VecQuantSpec:
    def __init__(self, quants_to_sizes: Dict[Enum, int]):
        self._quants_to_sizes = quants_to_sizes
        self._quants_to_idx = self._make_idx(quants_to_sizes)
        self._len = sum(quants_to_sizes.values())

    @classmethod
    def from_desc(cls, quant_desc: List[Enum], quants_to_sizes: Dict[Enum, int]):
        """Convenience method
        
        `quants_to_sizes.keys()` may be superset of `obs_desc`
        """
        return cls({q: quants_to_sizes[q] for q in quant_desc})
    
    def _make_idx(self, quants_to_sizes) -> dict:
        quants_to_idx = {}
        start = 0
        for q, size in quants_to_sizes.items():
            quants_to_idx[q] = np.s_[start: start + size]
            start += size
        return quants_to_idx

    def __len__(self):
        return self._len

    def __getitem__(self, name):
        return self._quants_to_idx[name]

    def __contains__(self, name):
        return name in self._quants_to_idx

    def __iter__(self):
        return iter(self._quants_to_idx)
    
    def make_normalizer(self, quants_to_max_values):
        max_vals = [quants_to_max_values[q] for q in self]
        repeats = [self._quants_to_sizes[q] for q in self]
        return np.repeat(max_vals, repeats)


class EnvSpec:
    def __init__(self,
                 env_name: str,
                 obs_spec: VecQuantSpec,
                 rfunc: Callable[..., float] = None):
        self.env_name = env_name
        self.obs_spec = obs_spec
        self.rfunc = rfunc
        
        self.obs_dim = len(self.obs_spec)
