from typing import Optional, Callable
import numpy as np
from sklearn.random_projection import SparseRandomProjection


def estimate_from_files(file_list: list, key: int, feat_dim: tuple, open_func: Callable, n_samples_estim: int,
                        n_components: Optional[int]):
    feats_for_estim = np.empty((n_samples_estim, *feat_dim))
    for i, file in enumerate(file_list[:n_samples_estim]):
        feats_for_estim[i, :] = open_func(file)[key].squeeze(0)
    if n_components:
        srp = SparseRandomProjection(n_components=n_components)
        srp.fit(feats_for_estim.reshape(n_samples_estim, -1))
    else:
        srp = SparseRandomProjection()
        sample_auto_proj = srp.fit_transform(feats_for_estim.reshape(n_samples_estim, -1))
        n_components = sample_auto_proj.shape[-1]
    return srp, n_components
