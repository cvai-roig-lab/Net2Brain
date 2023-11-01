
import warnings
from .netsetbase import NetSetBase
from .shared_functions import load_from_json
import timm
from typing import Callable
import torch.nn as nn
from timm.data import resolve_data_config
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.transforms_factory import create_transform as timm_create_transform
from torchvision import transforms as T
import torchextractor as tx
from PIL import Image
import os

class Timm(NetSetBase):

    def __init__(self, model_name, device):
        self.supported_data_types = ['image', 'video']
        self.netset_name = "timm"
        self.model_name = model_name
        self.device = device

        # Set config path:
        file_path = os.path.abspath(__file__)
        directory_path = os.path.dirname(file_path)
        self.config_path = os.path.join(directory_path, "architectures/configs/timm.json")


    def get_preprocessing_function(self, data_type):
        if data_type == 'image':
            return self.image_preprocessing
        elif data_type == 'video':
            warnings.warn("Models only support image-data. Will average video frames")
            return self.video_preprocessing
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

        # Load attributes from the json
        model_attributes = load_from_json(self.config_path, self.model_name)

        # Set the layers and model function from the attributes
        self.layers = model_attributes["nodes"]
    
        try:
            self.loaded_model = model_attributes["model_function"](self.model_name, pretrained=pretrained, features_only=True)
        except:
            self.loaded_model = model_attributes["model_function"](self.model_name, pretrained=pretrained)

        # Model to device
        self.loaded_model.to(self.device)

        # Randomize weights
        if not pretrained:
            self.loaded_model.apply(self.randomize_weights)

        # Put in eval mode
        self.loaded_model.eval()

    
    def image_preprocessing(self, image, model_name, device):
        preprocesser = self.create_preprocess()
        return preprocesser(image, model_name, device)
    
    def video_preprocessing(self, frame, model_name, device):
        preprocesser = self.create_preprocess()
        pil_frame = Image.fromarray(frame)
        return preprocesser(pil_frame, model_name, device)
    
    def clean_extracted_features(self, features):
        return features
    
    def extraction_function(self, data, layers_to_extract=None):

        self.layers = self.select_model_layers(layers_to_extract, self.layers, self.loaded_model)

        # Create a extractor instance
        self.extractor_model = tx.Extractor(self.loaded_model, self.layers)

        # Extract actual features
        _, features = self.extractor_model(data)

        return features

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




         
