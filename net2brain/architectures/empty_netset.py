import warnings
from .netsetbase import NetSetBase
from .shared_functions import load_from_json
import torchextractor as tx
import os



################################################
# This is a template for creating your netset. 
# Ideally you would only need to change
# the parts where it says "TODO:". 
# get_model() and extraction_function() should be good the way they are!
###############################################




class YOURNETSET(NetSetBase):  # Rename to your desired netset name

    def __init__(self, model_name, device):
        self.supported_data_types = ['image']  # TODO: Example data types
        self.netset_name = "MyCustomNetSet"
        self.model_name = model_name
        self.device = device

        # Set config path:
        file_path = os.path.abspath(__file__)
        directory_path = os.path.dirname(file_path)
        self.config_path = os.path.join(directory_path, "./") # TODO: Path to configuration file that lists all models & functions to access it (see other configs)


    def get_preprocessing_function(self, data_type):
        #TODO: If you have specific datatypes you might want to change the preprocessing.
        if data_type == 'image':
            return self.image_preprocessing
        else:
            raise ValueError(f"Unsupported data type for {self.netset_name}: {data_type}")
        

    def get_feature_cleaner(self, data_type):
        #TODO: If you have specific datatypes you might want to change the feauture cleaner
        if data_type == 'image':
            return YOURNETSET.clean_extracted_features
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

        return self.loaded_model


    def clean_extracted_features(self, features):
        # TODO: If you need to clean the extracted feautures you can do it here
        return features
    
    
    def extraction_function(self, data, layers_to_extract=None, model=None):

        self.layers = self.select_model_layers(layers_to_extract, self.layers, self.loaded_model)

        # Create a extractor instance
        self.extractor_model = tx.Extractor(self.loaded_model, self.layers)

        # Extract actual features
        _, features = self.extractor_model(data)

        return features

         
