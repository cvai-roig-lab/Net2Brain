import os

import torchextractor as tx

from .netsetbase import NetSetBase
from .shared_functions import load_from_json


class Audio(NetSetBase):

    def __init__(self, model_name, device):
        self.supported_data_types = ['audio']
        self.netset_name = "audio"
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.loader_kwargs = {}

        # Set config path:
        file_path = os.path.abspath(__file__)
        directory_path = os.path.dirname(file_path)
        self.config_path = os.path.join(directory_path, "configs/audio.json")

    def get_model(self, pretrained):
        # Load attributes from the json
        model_attributes = load_from_json(self.config_path, self.model_name)

        # Set the layers and model function from the attributes
        self.layers = model_attributes["nodes"]

        self.loaded_model = model_attributes["model_function"](self.model_name)

        if 'tokenizer' in model_attributes:
            # use the feature extractor
            self.tokenizer = model_attributes["tokenizer"](self.model_name)

            self.loader_kwargs = {
                'sample_rate': self.tokenizer.sampling_rate,
                'mono': getattr(self.tokenizer, 'mono', True),
            }
        else:
            # feature extraction is handled by the model
            self.loader_kwargs = {
                'sample_rate': getattr(self.loaded_model, 'sample_rate', None),
                'mono': getattr(self.loaded_model, 'mono', True),
            }

        # Model to device
        self.loaded_model.to(self.device)

        # Randomize weights
        if not pretrained:
            self.loaded_model.apply(self.randomize_weights)

        # Put in eval mode
        self.loaded_model.eval()

        return self.loaded_model

    def get_feature_cleaner(self, data_type):
        return Audio.clean_extracted_features

    def clean_extracted_features(self, features):
        cleaned_features = {}
        for layer, tensor in features.items():
            if isinstance(tensor, (tuple, list,)):
                tensor = tensor[0]

            # Check if the tensor requires grad, and detach if needed
            tensor = tensor.detach() if tensor.requires_grad else tensor

            cleaned_features[layer] = tensor.cpu()
        return cleaned_features

    def get_preprocessing_function(self, data_type):
        return self.audio_preprocessing

    def extraction_function(self, data, layers_to_extract=None):
        self.layers = self.select_model_layers(layers_to_extract, self.layers, self.loaded_model)

        # Create a extractor instance
        self.extractor_model = tx.Extractor(self.loaded_model, self.layers)

        if self.tokenizer is not None:
            data = self.tokenizer(data, sampling_rate=self.loader_kwargs['sample_rate'], return_tensors="pt")
            _, features = self.extractor_model(**data)
        else:
            # Extract actual features
            _, features = self.extractor_model(data)

        return features
