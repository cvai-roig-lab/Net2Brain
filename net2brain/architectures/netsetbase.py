from torchvision import transforms as trn
import torchextractor as tx
from PIL import Image
import cv2
import librosa
import torch
import torch.nn as nn
import warnings

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
            available_layers = tx.list_module_names(loaded_model)
            valid_layers = [layer for layer in layers_to_extract if layer in available_layers and layer != '']
            invalid_layers = set(layers_to_extract) - set(valid_layers)
            if invalid_layers:
                warnings.warn(f"Some layers are not present in the model and will not be extracted: {invalid_layers}. "
                            "Please call the 'layers_to_extract()' function from the FeatureExtractor to see all available layers.")
            return valid_layers
        elif network_layers:
            return [layer for layer in network_layers if layer != '']
        else:
            return [layer for layer in tx.list_module_names(loaded_model) if layer != '']



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
        """
        Args:
            image (Union[Image.Image, List[Image.Image]]): A PIL Image or a list of PIL Images.
            model_name (str): The name of the model, used to determine specific preprocessing if necessary.
            device (str): The device to which the tensor should be transferred ('cuda' for GPU, 'cpu' for CPU).

        Returns:
            Union[torch.Tensor, List[torch.Tensor]]: The preprocessed image(s) as PyTorch tensor(s).
        """
        transforms = trn.Compose([
            trn.Resize((224, 224)),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        img_tensor = transforms(image).unsqueeze(0)
        if device == 'cuda':
            img_tensor = img_tensor.cuda()
        return img_tensor


    def video_preprocessing(self, frame, model_name, device):
        # Convert numpy array to PIL Image
        pil_frame = Image.fromarray(frame)
        
        return NetSetBase.image_preprocessing(self, pil_frame, model_name, device)
    

    def text_preprocessing(self, text, model_name, device):
        return text

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
    
    
    def combine_image_data(self, feature_list):
        return feature_list[0]
    
    def combine_video_data(self, feature_list):
        """
        Averages the features extracted from multiple frames of a video.

        Args:
            feature_list (List[Dict[str, torch.Tensor]]): A list where each element is a dictionary. The keys of the
            dictionary are the layer names, and the values are the feature tensors extracted from a frame.

        Returns:
            Dict[str, torch.Tensor]: A dictionary where the keys are the layer names, and the values are the averaged 
            feature tensors across all frames.
        """
        # Initialize a dictionary to store the sum of features for each layer
        summed_features = {}

        for features in feature_list:
            for layer, data in features.items():
                if layer not in summed_features:
                    # If the layer is not in the dictionary, add it
                    summed_features[layer] = data.clone()
                else:
                    # If the layer is already in the dictionary, accumulate the features
                    summed_features[layer] += data

        # Calculate the average for each layer
        num_frames = len(feature_list)
        averaged_features = {layer: data / num_frames for layer, data in summed_features.items()}

        return averaged_features
    

    def combine_audio_data(self, feature_list):
        raise NotImplementedError
    
    def combine_text_data(self, feature_list):
        return feature_list[0]
    


    def load_image_data(self, data_path):
        return [Image.open(data_path).convert('RGB')]
    
    def load_video_data(self, data_path):
        # Logic to load video data using cv2
        # This will return a list of frames. Each frame is a numpy array.
        data_path = r"{}".format(data_path)  # Using raw string
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
    
    def load_text_data(self, data_path):
        """
        Load text data from a .txt file and return a list of sentences/words.

        Parameters:
        - data_path (str): Path to the .txt file.

        Returns:
        - list: List of sentences/words.
        """
        with open(data_path, 'r', encoding='utf-8') as file:
            text_data = file.read().splitlines()
        return text_data
    


    def randomize_weights(self, m):
        warnings.warn("Will initiate random weights")
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
