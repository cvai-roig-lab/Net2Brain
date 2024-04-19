from datetime import datetime
from pathlib import Path
from typing import Union, Optional, Callable, List

import torch
from tqdm.auto import tqdm

from .rdm import dist, valid_distance_functions
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
            chunk_size: int or None
                If not None, the RDM is created in chunks of the given size. This can be used to reduce the memory
                consumption.
            **kwargs: dict
                Additional keyword arguments for the distance function.
        """
        x = torch.flatten(x, start_dim=1)
        return dist(x, metric=distance, device=self.device, verbose=self.verbose, chunk_size=chunk_size, **kwargs)

    def create_rdms(self,
                    feature_path: Union[str, Path],
                    save_path: Optional[Union[str, Path]] = None,
                    save_format: RDMFileFormatType = 'pt',
                    distance: Union[str, Callable] = 'pearson',
                    chunk_size: Optional[int] = None,
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
            chunk_size: int or None
                If not None, the RDM is created in chunks of the given size. This can be used to reduce the memory
                consumption.
            **kwargs: dict
                Additional keyword arguments for the distance function. See rdm.dist
        """
        if save_path is None:
            save_path = Path.cwd() / 'rdms' / datetime.now().strftime('%d-%m-%y_%H-%M-%S')
        else:
            save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        iterator = FeatureIterator(feature_path)
        with tqdm(total=len(iterator), desc='Creating RDMs', disable=not self.verbose) as bar:
            for layer, stimuli, feats in iterator:
                feats = torch.from_numpy(feats).to(self.device)
                rdm_m = self._create_rdm(feats, distance=distance, chunk_size=chunk_size, **kwargs)
                meta = dict(distance=distance)

                rdm = LayerRDM(rdm=rdm_m, layer_name=layer, stimuli_name=stimuli, meta=meta)
                rdm.save(save_path, file_format=save_format)

                bar.update()
        return save_path
