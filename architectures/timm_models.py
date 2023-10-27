
import json
import warnings
import torchvision.models as models
from .netsetbase import NetSetBase
from .shared_functions import imagenet_preprocess, imagenet_preprocess_frames, torch_clean, load_from_json
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
import timm
import torch
from pathlib import Path
from PIL import Image
from typing import Union, Callable
import torch.nn as nn
from timm.data import resolve_data_config
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.transforms_factory import create_transform as timm_create_transform
from torchvision import transforms as T


class Timm(NetSetBase):

    def __init__(self, model_name):
        self.supported_data_types = ['image', 'video']
        self.netset_name = "timm"
        self.model_name = model_name


    def get_preprocessing_function(self, data_type):
        if data_type == 'image':
            return Timm.image_preprocessing
        elif data_type == 'video':
            warnings.warn("Models only support image-data. Will average video frames")
            return Timm.video_preprocessing
        else:
            raise ValueError(f"Unsupported data type for {self.netset_name}: {data_type}")
        

    def get_feature_cleaner(self, data_type):
        if data_type == 'image':
            return Timm.clean_extracted_features
        elif data_type == 'video':
            return Timm.clean_extracted_features
        else:
            raise ValueError(f"Unsupported data type for {self.netset_name}: {data_type}")
        

    def get_model(self, pretrained):

        # Set configuration path 
        config_path = "architectures/configs/timm.json"

        # Load attributes from the json
        model_attributes = load_from_json(config_path, self.model_name)

        # Set the layers and model function from the attributes
        self.layers = model_attributes["nodes"]
    
        try:
            self.loaded_model = model_attributes["model_function"](self.model_name, pretrained=pretrained, features_only=True)
        except:
            self.loaded_model = model_attributes["model_function"](self.model_name, pretrained=pretrained)

    
    def image_preprocessing(self, image, model_name, device):
        preprocesser = self.create_preprocess()
        return preprocesser(image, model_name, device)
    
    def video_preprocessing(self, frame, model_name, device):
        preprocesser = self.create_preprocess()
        return preprocesser(frame, model_name, device)
    
    def clean_extracted_features(self, features):
        return features
    
    def extraction_function(self, data, layers_to_extract=None):

        self.layers = self.select_model_layers(layers_to_extract, self.layers, self.loaded_model)

        # Create a extractor instance
        self.extractor_model = create_feature_extractor(self.loaded_model, return_nodes=self.layers)

        return self.extractor_model(data)

    def create_transform(self, model: nn.Module) -> T.Compose:
        """
        Creates a evaluation transform for the given TIMM model.
        """
        config = resolve_data_config({}, model=model, verbose=True)
        config = {
            "input_size": config.pop("input_size", (3, 224, 224)),
            "mean": config.pop("mean", IMAGENET_DEFAULT_MEAN),
            "std": config.pop("std", IMAGENET_DEFAULT_STD),
            "crop_pct": config.pop("crop_pct", 1.0),
            "interpolation": config.pop("interpolation", "bicubic"),
        }
        return timm_create_transform(**config, is_training=False)


    def create_preprocess(self) -> Callable:
        """
        Creates a preprocess function for the given TIMM model.
        """

        transform = self.create_transform(self.loaded_model)

        def preprocess(image, model_name, device):
            """Preprocesses image according to the networks needs

            Args:
                image (str/path): path to image
                model_name (str): name of the model (sometimes needes to differenciate between model settings)

            Returns:
                PIL-Image: Preprocesses PIL Image
            """

            image = transform(image).unsqueeze(0).to(device)
            return image

        return preprocess




         
