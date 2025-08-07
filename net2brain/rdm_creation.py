from datetime import datetime
from pathlib import Path
from typing import Union, Optional, Callable, List

import torch
from tqdm.auto import tqdm

from .rdm import dist, valid_distance_functions, standardize
from .rdm.feature_iterator import FeatureIterator
from .rdm.rdm import LayerRDM, RDMFileFormatType


class RDMCreator:
    """
    This class creates RDMs from the features that have been extracted with the feature extraction
    module
    """

    def __init__(self,
                 device: Union[str, torch.device] = 'cpu',
                 verbose: bool = False
                 ):
        """
        Args:
            device: str or torch.device
                The device to use for the RDM creation. If a string is given, it must be a valid device name.
            verbose: bool
                Whether to print progress bars or not.
        """
        self.device = torch.device(device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise ValueError(f"Device {self.device} is not available.")

        self.verbose = verbose

    @staticmethod
    def distance_functions() -> List[str]:
        return valid_distance_functions()

    def __call__(self, *args, **kwargs):  # TODO: implement as Pipeline?
        return

    def _create_rdm(self,
                    x: torch.Tensor,
                    distance: Union[str, Callable] = 'pearson',
                    standardize_on_dim: Optional[int] = None,
                    chunk_size: Optional[int] = None,
                    **kwargs) -> torch.Tensor:
        """
        Creates a RDM from the given features.

        Args:
            x: torch.Tensor
                The features to create the RDM from. The shape of the tensor must be (num_stimuli, *), where * is the
                shape of the feature vector that gets flattened.
            distance: str or callable
                The distance metric to use. If a string is given, it must be a valid distance function name. If a
                callable is passed, it must define a custom distance function.
            standardize_on_dim: int or None
                If not None, the features are standardized on the given dimension.
            chunk_size: int or None
                If not None, the RDM is created in chunks of the given size. This can be used to reduce the memory
                consumption.
            **kwargs: dict
                Additional keyword arguments for the distance function.
        """
        x = torch.flatten(x, start_dim=1)
        if standardize_on_dim is not None:
            x = standardize(x, dim=standardize_on_dim)
        return dist(x, metric=distance, device=self.device, verbose=self.verbose, chunk_size=chunk_size, **kwargs)

    def create_rdms(self,
                    feature_path: Union[str, Path],
                    save_path: Optional[Union[str, Path]] = None,
                    save_format: RDMFileFormatType = 'npz',
                    distance: Union[str, Callable] = 'pearson',
                    standardize_on_dim: Optional[int] = None,
                    chunk_size: Optional[int] = None,
                    multi_timepoint_rdms: Optional[str] = None,
                    dim_reduction: Optional[str] = None,
                    n_samples_estim: int = 100,
                    n_components: Optional[int] = 10000,
                    srp_before_pca: bool = False,
                    max_dim_allowed: Optional[int] = None,
                    **kwargs
                    ) -> Path:
        """
        Creates RDMs from the given features and saves them to the given path.

        Args:
            feature_path: str or Path
                Path to the directory containing the feature files.
            save_path: str or Path or None
                Path to the directory where the RDMs should be saved. If None, the RDMs are saved to a directory named
                `rdms` in the current working directory.
            save_format: str
                The format in which the RDMs should be saved. Choose from `pt` and `npz`.
            distance: str or callable
                The distance metric to use. If a string is given, it must be a valid distance function name. If a
                callable is passed, it must define a custom distance function.
            standardize_on_dim: int or None
                If not None, the features are standardized along the given dimension.
                I.e. for a two-dimensional tensor:
                    * dim=0: The mean and standard deviation are computed across rows (for each column)
                    * dim=1: The mean and standard deviation are computed across columns (for each row)
            chunk_size: int or None
                If not None, the RDM is created in chunks of the given size. This can be used to reduce the memory
                consumption.
            multi_timepoint_rdms: str or None, options are `clip` and `all_timepoints`
            dim_reduction: str or None
                Whether to apply dimensionality reduction to the features before creating the RDMs. Only supported
                when the features are *not* stored in a consolidated format. For consolidated storing of features,
                apply the dimensionality reduction at the feature extraction stage.
                Choose from `srp` (Sparse Random Projection) and `pca` (Principal Component Analysis).
                The next three parameters only apply when `dim_reduction` is not None.
            n_samples_estim: int
                The number of samples used for estimating the dimensionality reduction.
            n_components: int or None
                The number of components to reduce the features to. If None, the number of components is estimated.
                For PCA, `n_components` must be smaller than `n_samples_estim`.
            srp_before_pca (bool): Whether to apply Sparse Random Projection (SRP) before PCA. Use when features are so
                high-dimensional that PCA runs out of memory. Num of dims estimated by SRP.
            max_dim_allowed: int or None
                Optional: The threshold over which the dimensionality reduction is applied. If None, it is always
                applied.
            **kwargs: dict
                Additional keyword arguments for the distance function.
        """
        if save_path is None:
            save_path = Path.cwd() / 'rdms' / datetime.now().strftime('%d-%m-%y_%H-%M-%S')
        else:
            save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        iterator = FeatureIterator(feature_path,
                                   multi_timepoint_rdms=multi_timepoint_rdms,
                                   dim_reduction=dim_reduction,
                                   n_samples_estim=n_samples_estim,
                                   n_components=n_components,
                                   srp_before_pca=srp_before_pca,
                                   max_dim_allowed=max_dim_allowed)
        with tqdm(total=len(iterator), desc='Creating RDMs', disable=not self.verbose) as bar:
            for layer, stimuli, feats in iterator:
                feats = torch.from_numpy(feats).to(self.device)
                if multi_timepoint_rdms is not None:
                    clips = feats.shape[0]
                    rdm_c = []
                    for clip_idx in range(clips):
                        if multi_timepoint_rdms == 'clip':
                            rdm_t = self._create_rdm(feats[clip_idx], distance=distance, chunk_size=chunk_size,
                                                     standardize_on_dim=standardize_on_dim, **kwargs)
                        else:
                            timepoints = feats.shape[1]
                            rdm_t = []
                            for time_idx in range(timepoints):
                                rdm_ct = self._create_rdm(feats[clip_idx, time_idx], distance=distance, chunk_size=chunk_size,
                                                        standardize_on_dim=standardize_on_dim, **kwargs)
                                rdm_t.append(rdm_ct)
                            rdm_t = torch.stack(rdm_t, dim=0)
                        rdm_c.append(rdm_t)
                    rdm_m = torch.stack(rdm_c, dim=0)
                else:
                    rdm_m = self._create_rdm(feats, distance=distance, chunk_size=chunk_size,
                                                standardize_on_dim=standardize_on_dim, **kwargs)

                meta = dict(distance=distance, multi_timepoint_rdms=multi_timepoint_rdms)

                rdm = LayerRDM(rdm=rdm_m, layer_name=layer, stimuli_name=stimuli, meta=meta)
                rdm.save(save_path, file_format=save_format)

                bar.update()
        return save_path
