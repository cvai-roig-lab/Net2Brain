import warnings
from .netsetbase import NetSetBase
from .shared_functions import load_from_json
import torchextractor as tx
import os

class Standard(NetSetBase):

    def __init__(self, model_name, device):
        self.supported_data_types = ['image', 'video']
        self.netset_name = "Standard"
        self.model_name = model_name
        self.device = device

        # Set config path:
        file_path = os.path.abspath(__file__)
        directory_path = os.path.dirname(file_path)
        self.config_path = os.path.join(directory_path, "architectures/configs/pytorch.json")


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
            return Standard.clean_extracted_features
        elif data_type == 'video':
            return Standard.clean_extracted_features
        else:
            raise ValueError(f"Unsupported data type for {self.netset_name}: {data_type}")
        



    def get_model(self, pretrained):

        # Load attributes from the json
        model_attributes = load_from_json(self.config_path, self.model_name)

        # Set the layers and model function from the attributes
        self.layers = model_attributes["nodes"]
        self.loaded_model = model_attributes["model_function"](pretrained=pretrained)

        # Model to device
        self.loaded_model.to(self.device)

        # Randomize weights
        if not pretrained:
            self.loaded_model.apply(self.randomize_weights)

        # Put in eval mode
        self.loaded_model.eval()


    def clean_extracted_features(self, features):
        return features
    
    
    def extraction_function(self, data, layers_to_extract=None):

        self.layers = self.select_model_layers(layers_to_extract, self.layers, self.loaded_model)

        # Create a extractor instance
        self.extractor_model = tx.Extractor(self.loaded_model, self.layers)

        # Extract actual features
        _, features = self.extractor_model(data)

        return features

         
