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
                    consolidated: bool = False,
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
            **kwargs: dict
                Additional keyword arguments for the distance function.
        """
        if save_path is None:
            save_path = Path.cwd() / 'rdms' / datetime.now().strftime('%d-%m-%y_%H-%M-%S')
        else:
            save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        iterator = FeatureIterator(feature_path)
        if consolidated:  # SET MANUALLY BECAUSE AUTODETECTION IS WRONG!!!
            iterator.format = FeatureFormat.NPZ_CONSOLIDATED
        else:
            iterator.format = FeatureFormat.NPZ_SEPARATE
        iterator.engine = engine_registry.get_engine(iterator.format)(iterator.root)
        with tqdm(total=len(iterator), desc='Creating RDMs', disable=not self.verbose) as bar:
            for layer, stimuli, feats in iterator:
                feats = torch.from_numpy(feats).to(self.device)
                rdm_m = self._create_rdm(feats, distance=distance, chunk_size=chunk_size,
                                         standardize_on_dim=standardize_on_dim, **kwargs)
                meta = dict(distance=distance)

                rdm = LayerRDM(rdm=rdm_m, layer_name=layer, stimuli_name=stimuli, meta=meta)
                rdm.save(save_path, file_format=save_format)
                del feats
                del rdm_m
                del rdm

                bar.update()
        return save_path
