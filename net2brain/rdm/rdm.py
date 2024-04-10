from pathlib import Path
from typing import List, Union, Optional, Dict, Literal
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from .dist_utils import to_distance_matrix, to_condensed, is_condensed_1d

RDMFileFormat = ["pt", "npz"]
RDMFileFormatType = Literal["pt", "npz"]


class RDM:
    """
    Container for a RDM. Abstracts away the difference between condensed and square RDMs.
    """

    def __init__(self,
                 rdm: Union[torch.Tensor, np.ndarray]):
        """
        Args:
            rdm: torch.Tensor or np.ndarray
                The RDM. Must be a square or condensed matrix.
        """
        if not isinstance(rdm, (np.ndarray, torch.Tensor)):
            raise ValueError(f"Incompatible type {type(rdm)} for RDM. Must be np.ndarray or torch.Tensor.")

        self.rdm = torch.from_numpy(rdm) if isinstance(rdm, np.ndarray) else rdm
        self.is_condensed = is_condensed_1d(self.rdm)

    def to_vector(self):
        """
        Converts the RDM to a condensed vector.
        """
        if not self.is_condensed:
            self.is_condensed = True
            self.rdm = to_condensed(self.rdm)

    def to_matrix(self):
        """
        Converts the RDM to a square matrix.
        """
        if self.is_condensed:
            self.is_condensed = False
            self.rdm = to_distance_matrix(self.rdm)


class LayerRDM(RDM):
    """
    Container for a RDM of a single layer with additional meta data.

    Args:
        rdm: torch.Tensor or np.ndarray
            The RDM of the layer. Must be a square or condensed matrix.
        layer_name: str or None
            The name of the layer.
        stimuli_name: List[str] or None
            The names of the stimuli.
        meta: dict or None
            Additional meta data.
    """

    def __init__(self,
                 rdm: Union[torch.Tensor, np.ndarray],
                 layer_name: Optional[str] = None,
                 stimuli_name: Optional[List[str]] = None,
                 meta: Optional[Dict] = None):
        super().__init__(rdm)
        self.layer_name = layer_name
        self.stimuli_name = stimuli_name
        self.meta = meta

    def __len__(self) -> int:
        return len(self.layer_name)

    def __repr__(self) -> str:
        return f"LayerRDM(layer={self.layer_name}, num_stimuli={len(self)}, condensed={self.is_condensed})"

    def __str__(self) -> str:
        return f"RDM for layer {self.layer_name} with {len(self)} stimuli"

    def __eq__(self, other) -> bool:
        return self.rdm == other.rdm

    def save(self, path: Union[str, Path], file_format: RDMFileFormatType = "pt", force: bool = True) -> None:
        """
        Save the RDM to file. Always saves the RDM as condensed to save space.

        Args:
            path: str or Path
                Path to the file where the RDM should be saved. If a directory is given, the RDM is saved as
                `RDM_{layer_name}.{file_format}` in the given directory. If a file is given, the RDM is saved to this
                file.
            file_format: str
                The format in which the RDM should be saved. Choose from `pt` and `npz`.
            force: bool

        """
        # always save as condensed to save space
        rdm = to_condensed(self.rdm) if not self.is_condensed else self.rdm
        # convert to cpu if necessary
        rdm = rdm.cpu()
        # create data dict including meta data
        data = dict(rdm=rdm, stimuli_name=self.stimuli_name, layer_name=self.layer_name, meta=self.meta)

        path = Path(path)
        if path.is_dir():
            layer_name = self.layer_name.replace(".", "_") if self.layer_name is not None else uuid4().hex
            path = Path(path) / f"RDM_{layer_name}.{file_format}"
        elif path.exists():
            raise FileExistsError(f"File {path} already exists. Cannot save RDM.")
        else:
            path = path.with_suffix(f".{file_format}")

        if file_format == "pt":
            torch.save(data, path)
        elif file_format == "npz":
            np.savez(path, **data)
        else:
            s = f"Save format `{file_format}` not supported. Choose from {RDMFileFormat}"
            raise ValueError(s)

    @classmethod
    def from_file(cls, path: Union[str, Path]):
        """
        Load a RDM from file.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File {path} does not exist.")
        if path.suffix == ".pt":
            data = torch.load(path)
        elif path.suffix == ".npz":
            data = np.load(path, allow_pickle=True)
            # convert numpy object array to python object
            data = dict(map(lambda x: (x[0], x[1].item() if x[1].ndim == 0 else x[1]), data.items()))
        else:
            s = f"File format `{path.suffix}` not supported. Choose from {RDMFileFormat}"
            raise ValueError(s)

        return cls(**data)

    def plot(self,
             indices: List[int] = None,
             **kwargs):
        """
        Plot the RDM as a heatmap.

        Args:
            indices: List[int]
                The indices of the stimuli to plot. If None, all stimuli are plotted.
            **kwargs: dict
                Additional keyword arguments for the heatmap.
        """
        if self.is_condensed:
            feats = to_distance_matrix(self.rdm)
        else:
            feats = self.rdm

        stimuli_name = self.stimuli_name
        if indices is not None:
            feats = torch.index_select(feats, 0, torch.tensor(indices))
            feats = torch.index_select(feats, 1, torch.tensor(indices))
            stimuli_name = [self.stimuli_name[i] for i in indices]

        if 'distance' in self.meta.keys():
            cbar_kws = {'label': self.meta['distance']}
            kwargs['cbar_kws'] = cbar_kws if 'cbar_kws' not in kwargs.keys() else {**kwargs['cbar_kws'], **cbar_kws}

        ax = sns.heatmap(
            data=feats,
            xticklabels=stimuli_name,
            yticklabels=stimuli_name,
            **kwargs
        )
        if self.layer_name is not None:
            ax.set_title(f"Layer={self.layer_name}")

        plt.show()

        return ax
