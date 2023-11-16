import warnings
from .netsetbase import NetSetBase
from .shared_functions import load_from_json
import torchextractor as tx
import os
from transformers import AutoTokenizer, BartModel, AlignTextModel, AlbertModel
import transformers
import torch

class Huggingface(NetSetBase):

    def __init__(self, model_name, device):
        self.supported_data_types = ['text']
        self.netset_name = "Huggingface"
        self.model_name = model_name
        self.device = device

        # Set config path:
        file_path = os.path.abspath(__file__)
        directory_path = os.path.dirname(file_path)
        self.config_path = os.path.join(directory_path, "configs/huggingface.json")


    def get_preprocessing_function(self, data_type):
        if data_type == 'text':
            return self.text_preprocessing
        else:
            raise ValueError(f"Unsupported data type for {self.netset_name}: {data_type}")
        
    def get_feature_cleaner(self, data_type):
        if data_type == 'text':
            return Huggingface.clean_extracted_features
        else:
            raise ValueError(f"Unsupported data type for {self.netset_name}: {data_type}")
        



    def get_model(self, pretrained):

        # Load attributes from the json
        model_attributes = load_from_json(self.config_path, self.model_name)

        # Set the layers and model function from the attributes
        self.layers = model_attributes["nodes"]
        self.loaded_model = model_attributes["model_function"](self.model_name)

        # Model to device
        self.loaded_model.to(self.device)

        # Randomize weights
        if not pretrained:
            self.loaded_model.apply(self.randomize_weights)

        # Put in eval mode
        self.loaded_model.eval()


    def clean_extracted_features(self, features):
        """
        Clean extracted features to ensure tensors are not stored in a tuple.

        Parameters:
        - features (dict): Dictionary of layers with corresponding torch tensors.

        Returns:
        - dict: Cleaned features.
        """
        cleaned_features = {}
        for layer, tensor_tuple in features.items():
            # Unpack the tuple and extract the tensor
            tensor = tensor_tuple[0]
            # Check if the tensor requires grad, and detach if needed
            tensor = tensor.detach() if tensor.requires_grad else tensor
            # Convert to CPU and numpy
            tensor_numpy = tensor.cpu().numpy()
            cleaned_features[layer] = torch.tensor(tensor_numpy)
        return cleaned_features

    
    
    def extraction_function(self, data, layers_to_extract=None, model=None):

        self.layers = self.select_model_layers(layers_to_extract, self.layers, self.loaded_model)

        # Tokenizer for text
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        inputs = tokenizer(data, return_tensors="pt")

        # Create a extractor instance
        self.extractor_model = tx.Extractor(self.loaded_model, self.layers)

        # Extract actual features
        _, features = self.extractor_model(**inputs)

        return features

         
