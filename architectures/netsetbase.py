
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
from .shared_functions import imagenet_preprocess, imagenet_preprocess_frames, torch_clean, load_from_json
import torchextractor as tx
from PIL import Image
import cv2
import librosa
import torch
import torch.nn as nn

# Base class for all NetSets
class NetSetBase:
    # Class-level dictionary to hold all registered NetSet classes
    _registry = {}

    # Attributes to be set by subclasses
    supported_data_types = []  # List of supported data types for each netset
    netset_name = None  # Name of the netset
    model_name = None  # Model name used within the netset
    layers = None  # Layers in the model to be used for feature extraction
    loaded_model = None  # The loaded model instance
    extractor_model = None  # The feature extractor model instance
    device = None # Device for computation

    @classmethod
    def register_netset(cls):
        cls._registry[cls.__name__] = cls

    @classmethod
    def initialize_netset(cls, model_name, netset_name, device):
        # Return an instance of the netset class based on the netset_name from the registry
        if netset_name in cls._registry:
            return cls._registry[netset_name](model_name, device)
        else:
            raise ValueError(f"Unknown netset: {netset_name}")

    @classmethod
    def supports_data_type(cls, data_type):
        return data_type in cls.supported_data_types
    
    def select_model_layers(self, layers_to_extract, network_layers, loaded_model):
        if layers_to_extract:
            return layers_to_extract
        elif network_layers:
            return network_layers
        else:
            return tx.list_module_names(loaded_model)


    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.register_netset()

    # Placeholder methods that should be implemented by subclasses
    def get_preprocessing_function(self, data_type):
        raise NotImplementedError

    def get_feature_cleaner(self, data_type):
        raise NotImplementedError

    def get_model(self, pretrained):

        """
        # Set configuration path 
        config_path = "architectures\configs\pytorch.json"

        # Load model and layers from the json
        model_function, self.layers = load_from_json(config_path, self.model_name)

        # Inititate the model
        self.loaded_model = model_function(pretrained=pretrained)
        """

        raise NotImplementedError

    def image_preprocessing(self, image, model_name, device):
        raise NotImplementedError

    def video_preprocessing(self, frame, model_name, device):
        raise NotImplementedError

    def clean_extracted_features(self, features):
        # return features
        raise NotImplementedError

    def extraction_function(self, data, layers_to_extract=None):
        
        """
        # Which layers to extract
        self.layers = self.select_model_layers(layers_to_extract, self.layers, self.loaded_model)

        # Create a extractor instance
        self.extractor_model = create_feature_extractor(self.loaded_model, return_nodes=self.layers)
        """

        raise NotImplementedError
    
    
    def load_image_data(self, data_path):
        return Image.open(data_path).convert('RGB')
    
    def load_video_data(self, data_path):
        # Logic to load video data using cv2
        # This will return a list of frames. Each frame is a numpy array.
        cap = cv2.VideoCapture(data_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames
    
    def load_audio_data(self, data_path):
        # Logic to load audio data using librosa
        # This returns a numpy array representing the audio and its sample rate
        y, sr = librosa.load(data_path, sr=None)
        return y, sr
    

    def randomize_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
