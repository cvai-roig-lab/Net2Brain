from collections import defaultdict
from datetime import datetime
import os.path as op
import os
from pathlib import Path
from PIL import Image

import numpy as np
from rsatoolbox.data.dataset import Dataset
import torch
import torch.nn as nn
from torch.autograd import Variable as V
import torchextractor as tx
from torchvision import transforms as T
from tqdm import tqdm

import net2brain.architectures.pytorch_models as pymodule
import net2brain.architectures.slowfast_models as pyvideo
import net2brain.architectures.taskonomy_models as taskonomy
import net2brain.architectures.timm_models as timm
import net2brain.architectures.torchhub_models as torchmodule
import net2brain.architectures.unet_models as unet
import net2brain.architectures.yolo_models as yolo
import net2brain.architectures.toolbox_models as toolbox_models
import net2brain.architectures.cornet_models as cornet_models

## Get available networks
AVAILABLE_NETWORKS = {
    'standard': list(pymodule.MODELS.keys()),
    'toolbox': list(toolbox_models.MODELS.keys()),
    'timm': list(timm.MODELS.keys()),
    'cornet': list(cornet_models.MODELS.keys()),
    'pytorch': list(torchmodule.MODELS.keys()),
    'unet': list(unet.MODELS.keys()),
    'taskonomy': list(taskonomy.MODELS.keys()),
    'pyvideo': list(pyvideo.MODELS.keys())
}

## TODO: don't import unless needed

try:
    #import clip
    import net2brain.architectures.clip_models as clip_models
    AVAILABLE_NETWORKS.update({'clip': list(clip_models.MODELS.keys())})
except ModuleNotFoundError:
    print("Clip models are not installed.")
    clip_exist = False

try:
    #import vissl
    import net2brain.architectures.vissl_models as vissl_models
    AVAILABLE_NETWORKS.update({'vissl': list(vissl_models.MODELS.keys())})
except ModuleNotFoundError:
    print("vissl models are not installed")
    vissl_exist = False

try:
    import detectron2
    import net2brain.architectures.detectron2_models as detectron2_models
    AVAILABLE_NETWORKS.update(
        {'detectron2': list(detectron2_models.MODELS.keys())}
    )
except ModuleNotFoundError:
    print("detectron2 is not installed.")
    detectron_exist = False


## Define relevant paths
CURRENT_DIR = op.abspath(os.curdir)
BASE_DIR = op.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT_DIR = op.dirname(BASE_DIR)  # path to parent folder
FEATURES_DIR = op.join(PARENT_DIR, 'features')
GUI_DIR = op.join(BASE_DIR, 'helper', 'gui')
INPUTS_DIR = op.join(PARENT_DIR, 'input_data')
STIMULI_DIR = op.join(INPUTS_DIR, 'stimuli_data')
RDMS_DIR = op.join(PARENT_DIR, 'rdms')
BRAIN_DIR = op.join(INPUTS_DIR, 'brain_data')


def print_all_models():
    """Returns available models.

    Returns
    -------
    dict
        Available models by netset.
    """
    print("\n")
    for key, values in AVAILABLE_NETWORKS.items():
        print(f"NetSet: {key}")
        print(f"Models: {[v for v in values]}")
        print("\n")
    return


def print_all_netsets():
    """Returns available netsets.

    Returns
    -------
    list
       Available netsets.
    """
    return list(AVAILABLE_NETWORKS.keys())


def print_netset_models(netset):
    """Returns available models of a given netset.

    Parameters
    ----------
    netset : str
        Name of netset.

    Returns
    -------
    list
        Available models.

    Raises
    ------
    KeyError
        If netset is not available in the toolbox.
    """
    if netset in list(AVAILABLE_NETWORKS.keys()):
        return AVAILABLE_NETWORKS[netset]
    else:
        raise KeyError(
            f"This netset '{netset}' is not available. Available netsets are", 
            list(AVAILABLE_NETWORKS.keys())
        )


def find_model_like(name):
    """Find models containing the given string. Way of finding a model within \
        the model zoo.

    Parameters
    ----------
    name : str
        Name models.
    """
    for key, values in AVAILABLE_NETWORKS.items():
        for model_names in values:
            if name.lower() in model_names.lower():
                print(f'{key}: {model_names}')
                





def randomize_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform(m.weight)





class FeatureExtractor:
    # self.layers_to_extract= The layers we want to extract
    ## TODO: define this here for all types or dont

    def __init__(
        self, model, netset=None, layers_to_extract=None, device='cpu', 
        transforms=None, pretrained=True
    ):
        """Initializes feature extractor.

        Parameters
        ----------
        model : str or PyTorch model.
            If string is provided, the model will be loaded from the model zoo.
            Else the custom model will be used.
        netset : str optional
            NetSet from which to extract model, by default None.
        layers_to_extract : list, optional
            List of layers to extract the features from. If None, use default
            layers.   
        device : str
            CPU or CUDA.
        transforms : Pytorch Transforms, optional
            The transforms to be applied to the inputs, by default None.
        """
        # Set model and device
        self.device = device
        self.pretrained = pretrained
        
        # Load model from netset or load custom model
        if type(model) == str:
            if netset == None:
                raise NameError("netset must be specified")
            self.load_netset_model(model, netset, layers_to_extract)
        else: 
            self.load_model(model, layers_to_extract, transforms)

    def load_model(self, model, layers_to_extract, transforms=None):
        """Load a custom model.

        Parameters
        ----------
        model : PyTorch model
            Custom model.
        transforms : PyTorch Transforms, optional
             The transforms to be applied to the inputs, by default None.
        """
        self.model = model
        self.model.to(self.device)
        self.model_name = "Custom model"
        
        # Define preprocessing strategy
        self.transforms = transforms
        self.preprocess = self.preprocess_image
        
        # Define feature extraction parameters
        self.layers_to_extract = layers_to_extract
        self.features_path = None
        self._extractor = self._extract_features_tx
        self._features_cleaner = self._no_clean

    def load_netset_model(self, model_name, netset, layers_to_extract):
        """Load a model from the model zoo.

        Parameters
        ----------
        model_name : str
            Name of the model.
        netset : str
            Netset from which to extract the model.
        """
        self.model_name = model_name

        # Some Torchversion return a download error on MacOS, this is a fix
        torch.hub._validate_not_a_forked_repo=lambda a,b,c: True


        print(self.pretrained)

        if netset == "standard":
            self.module = pymodule
            self.model = self.module.MODELS[model_name](pretrained=self.pretrained)
            self._extractor = self._extract_features_tx
            self._features_cleaner = self._no_clean

        elif netset == 'pytorch':
            self.module = torchmodule
            self.model = self.module.MODELS[model_name](
                'pytorch/vision:v0.10.0', self.model_name, pretrained=self.pretrained
            )

            if not self.pretrained:
                self.model.to(self.device)
                self.model.apply(randomize_weights)

            self.model.eval()
            self._extractor = self._extract_features_tx
            self._features_cleaner = self._torch_clean


        elif netset == 'toolbox':
            self.module = toolbox_models
            self.model = self.module.MODELS[model_name](pretrained=self.pretrained)
            if not self.pretrained:
                self.model.to(self.device)
                self.model.apply(randomize_weights)
            self.model.eval()
            self._extractor = self._extract_features_tx
            self._features_cleaner = self._torch_clean

        elif netset == 'taskonomy':
            self.module = taskonomy
            self.model = self.module.MODELS[model_name](eval_only=True)
            if self.pretrained:
                checkpoint = torch.utils.model_zoo.load_url(
                    self.module.MODEL_WEIGHTS[model_name]
                ) # Load weights
                self.model.load_state_dict(checkpoint['state_dict'])
            self._extractor = self._extract_features_tx
            self._features_cleaner = self._no_clean

        elif netset == 'unet':
            self.module = unet
            self.model = self.module.MODELS[model_name](
                'mateuszbuda/brain-segmentation-pytorch', self.model_name, 
                in_channels=3, out_channels=1, init_features=32, 
                pretrained=self.pretrained
            )
            if not self.pretrained:
                self.model.to(self.device)
                self.model.apply(randomize_weights)
            self._extractor = self._extract_features_tx
            self._features_cleaner = self._no_clean

        elif netset == 'clip':
            self.module = clip_models
            correct_model_name = self.model_name.replace("_-_", "/")
            self.model = self.module.MODELS[model_name](
                correct_model_name, device=self.device
            )[0]

            if not self.pretrained:
                self.model.to(self.device)
                self.model.apply(randomize_weights)

            self._extractor = self._extract_features_tx_clip
            self._features_cleaner = self._no_clean

        elif netset == 'cornet':
            self.module = cornet_models
            self.model = self.module.MODELS[model_name](pretrained=self.pretrained)
            self.model = torch.nn.DataParallel(self.model)
            self._extractor = self._extract_features_tx
            self._features_cleaner = self._CORnet_RT_clean

        elif netset == 'yolo':
            # TODO: ONLY WORKS ON CUDA YET - NEEDS CLEANUP
            self.module = yolo
            self.model = self.module.MODELS[model_name](
                'ultralytics/yolov5', 'yolov5l', pretrained=self.pretrained, 
                device=self.device
            )
            self._extractor = self._extract_features_tx
            self._features_cleaner = self._no_clean

        elif netset == 'detectron2':
            self.module = detectron2_models
            config = self.module.configurator(self.model_name)
            self.model = self.module.MODELS[model_name](config)
            if not self.pretrained:
                self.model.to(self.device)
                self.model.apply(randomize_weights)
            self.model.eval()
            self._extractor = self._extract_features_tx
            self._features_cleaner = self._detectron_clean

        elif netset == 'vissl':
            self.module = vissl_models
            config = self.module.configurator(self.model_name)
            self.model = (
                self.module.MODELS[model_name]
                (config.MODEL, config.OPTIMIZER)
            )
            if not self.pretrained:
                self.model.to(self.device)
                self.model.apply(randomize_weights)

            self._extractor = self._extract_features_tx
            self._features_cleaner = self._no_clean

        elif netset == "timm":
            self.module = timm
            try:
                self.model = self.module.MODELS[model_name](
                    model_name, pretrained=self.pretrained, features_only=True)
            except:
                self.model = self.module.MODELS[model_name](
                    model_name, pretrained=self.pretrained)
            # Handle layers to extract differently
            if layers_to_extract == None:
                self.layers_to_extract = self.module.MODEL_NODES[model_name]
                if self.layers_to_extract == []:
                    self._extractor = self._extract_features_timm
                else:
                    self._extractor = self._extract_features_tx
            self._features_cleaner = self._no_clean

        elif netset == 'pyvideo':
            self.module = pyvideo
            self.model = self.module.MODELS[model_name](
                'facebookresearch/pytorchvideo', self.model_name, 
                pretrained=self.pretrained
            )
            if not self.pretrained:
                self.model.to(self.device)
                self.model.apply(randomize_weights)

            self.model.eval()
            self._extractor = self._extract_features_tx
            self._features_cleaner = self._slowfast_clean

        else:
            raise NotImplementedError(f"The netset '{netset}' does not appear to be implement. Perhaps check spelling!")

        # Define layers to extract
        if (layers_to_extract == None) and (netset != "timm"):
            self.layers_to_extract = self.module.MODEL_NODES[model_name]
        elif netset!= "timm":
            self.layers_to_extract = layers_to_extract

        # Send model to device
        self.model.to(self.device)

        # Define standard preprocessing
        self.preprocess = self.module.preprocess

    def preprocess_image(self, image, model_name):
        """Default preprocessing based on ImageNet standard training.

        Parameters
        ----------
        image : str
            Path to the image to be preprocessed.

        Returns
        -------
        PyTorch Tensor
            Preprocessed image.
        """
        if self.transforms is None:
            self.transforms = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        image = Image.open(image).convert('RGB')
        image = V(self.transforms(image).unsqueeze(0))
        image = image.to(self.device)
        return image

    def _extract_features_tx(self, image):
        """Extract features with torch extractor.

        Parameters
        ----------
        image : Torch Tensor
            Preprocessed image.

        Returns
        -------
        dict of Torch Tensors
            Features by layer.
        """
        extractor = tx.Extractor(self.model, self.layers_to_extract)
        _, features = extractor(image)
        features = self._features_cleaner(features)
        return features

    def _extract_features_tx_clip(self, image):
        """Extract CLIP features with torch extractor.

        Parameters
        ----------
        image : Torch Tensor
            Preprocessed image.

        Returns
        -------
        dict of Torch Tensors
            Features by layer.
        """
        extractor = tx.Extractor(self.model, self.layers_to_extract)
        image_data = image[0]
        tokenized_data = image[1]
        _, features = extractor(image_data, tokenized_data)
        features = self._features_cleaner(features)
        return features

    def _extract_features_timm(self, image):
        """Extract features with timm.

        Parameters
        ----------
        image : Torch Tensor
            Preprocessed image.

        Returns
        -------
        dict of Torch Tensors
            Features by layer.
        """
        features = self.model(image)
        # Convert the features into a dict because timm extractor returns a 
        # list of tensors
        converted_features = {}
        for counter, feature in enumerate(features):
            converted_features[f"feature {str(counter+1)}"] = feature.data.cpu()
        features = self._features_cleaner(converted_features)
        return converted_features

    def _no_clean(self, features):
        """Cleanup after feature extraction: This one requires no cleanup.
        Just put it on cpu in case it isn't yet!

        Args:
            features (dict:tensors): dictionary of tensors

        Returns:
            (dict:tensors): dictionary of tensors
        """
        return {key: value.data.cpu() for key, value in features.items()}

    def _torch_clean(self, features):
        """Cleanup function after feature extraction: 
        This one contains subdictionaries which need to be eliminated.

        Args:
            features (dict:tensors): dictionary of tensors

        Returns:
            (dict:tensors): dictionary of tensors
        """
        new_features = {}
        for key, value in features.items():
            try:
                new_features[key] = value["out"].data.cpu()
            except:
                new_features[key] = value.data.cpu()
        return new_features

    def _detectron_clean(self, features):
        """Cleanup function after feature extraction.
        Detectron models contain subdictionaries which need to be eliminated.

        Args:
            features (dict:tensors): dictionary of tensors

        Returns:
            (dict:tensors): dictionary of tensors
        """
        clean_dict = {}
        for key, subdict in features.items():
            keys = list(subdict.keys())
            for key in keys:
                clean_dict.update({key: subdict[key].cpu()})
        return clean_dict

    def _CORnet_RT_clean(self, features):
        """Cleanup function after feature extraction.
        The RT-Model contains subdirectories.

        Args:
            features (dict:tensors): dictionary of tensors

        Returns:
            (dict:tensors): dictionary of tensors
        """

        if self.model_name == "cornet_rt":
            clean_dict = {}
            for A_key, subtuple in features.items():
                keys = [A_key + "_A", A_key + "_B"]
                for counter, key in enumerate(keys):
                    clean_dict.update({key: subtuple[counter].cpu()})
                    break  # we actually only want one key
            return clean_dict
        else:
            return {key: value.cpu() for key, value in features.items()}

    def _slowfast_clean(self, features):
        """Cleanup function after feature extraction.
        Some features have two values (list).

        Args:
            features (dict:tensors): dictionary of tensors

        Returns:
            (dict:tensors): dictionary of tensors
        """

        clean_dict = {}
        for A_key, subtuple in features.items():
            keys = [A_key + "_slow", A_key + "_fast"]

            try:  # if subdict is a list of two values
                for counter, key in enumerate(keys):
                    clean_dict.update({key: subtuple[counter].cpu()})
            except:
                clean_dict.update({A_key: subtuple.cpu()})

        return clean_dict


    def extract(
        self, dataset_path, save_format='npz', save_path=None, 
    ):
        """Compute feature extraction from image dataset.

        Parameters
        ----------
        dataset_path : str or pathlib.Path
            Path to the images to extract the features from. Images cneed to be
            .jpg or .png.
        save_format : str, optional
            Format to save the features in. Can be 'npz', 'pt' or 'datasaet',
            by default 'npz'. If 'dataset', the features are saved in the
            Dataset class format of the rsa toolbox.
        save_path : str or pathlib.Path, optional
            Path to save the features to. If None, the folder where the
            features are saved is named after the current date in the 
            format "{year}_{month}_{day}_{hour}_{minute}".    
        
        """
        # Define save parameters
        self.save_format = save_format
        if save_path is None:
            self.save_path = create_save_path()
        else:
            self.save_path = Path(save_path)
            self.save_path.mkdir(parents=True, exist_ok=True)

        # Find all input files
        image_files = [
            i for i in Path(dataset_path).iterdir() 
            if i.suffix in ['.jpeg', '.jpg', '.png']
        ]
        image_files.sort()

        # Extract features from images
        if image_files != []:
            fts = self._extract_from_images(image_files)
        else:
            raise ValueError(
                "Could not find any .jpg or .png images in the given folder."
            )

        return fts


    def _extract_from_images(self, image_files):
        ## TODO: check no weird network names for saving
        
        if self.save_format == 'dataset':
            all_fts = defaultdict(list)

        for img in tqdm(image_files):
            
            # Preprocess image and extract features
            processsed_img = self.preprocess(img, self.model_name)
            fts = self._extractor(processsed_img)

            # Save features if npz or pt
            if self.save_format == 'npz':
                fts = {k: v.detach().numpy() for k, v in fts.items()}
                filename = self.save_path / f'{self.model_name}_{img.stem}.npz'
                np.savez(filename, **fts)
            elif self.save_format == 'pt':
                filename = self.save_path / f'{self.model_name}_{img.stem}.pt'
                torch.save(fts, filename)
            # Add features to dictionary if dataset
            elif self.save_format == 'dataset':
                for l in fts.keys():
                    all_fts[l].append(fts[l])

        # Save and return features per layer in rsa toolbox format 
        if self.save_format == 'dataset':
            obs_imgs = {'images': np.array([i.stem for i in image_files])}
            fts_datasets = {}
            for l in all_fts.keys():
                d = torch.flatten(torch.stack(all_fts[l]), start_dim=1)
                fts_datasets[l] = Dataset(
                    measurements=d.detach().numpy(),
                    descriptors = {'dnn': self.model_name, 'layer': l},
                    obs_descriptors = obs_imgs
                )
                filename = self.save_path / f'{self.model_name}_{l}.hdf5'
                fts_datasets[l].save(filename)
            return fts_datasets
        else:
            return


    def get_all_layers(self):
        """Helping function to extract all possible layers from a model

        Returns:
            list: all layers we can extract features from
        """
        layers = tx.list_module_names(self.model)
        return layers


def create_save_path():
    """ Creates folder to save the image features.

    Returns
    -------
    pathlib Path
        Path to directory of features. Named after the current date in the
        format "{year}_{month}_{day}_{hour}_{minute}"
    """
    # Get current time and format string accordingly
    now = datetime.now()
    now_formatted = f'{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}'

    # Create directory
    save_path = Path(f"features/{now_formatted}")
    save_path.mkdir(parents=True, exist_ok=True)

    return save_path
