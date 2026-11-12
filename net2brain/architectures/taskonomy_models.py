
import warnings
from .netsetbase import NetSetBase, create_extractor
from torchvision import transforms as trn
from .shared_functions import load_from_json
from .implemented_models import taskonomy_model
import torch
import numpy as np
from skimage.color import rgb2lab
from PIL import Image
import os


class Taskonomy(NetSetBase):

    def __init__(self, model_name, device):
        self.supported_data_types = ['image', 'video']
        self.netset_name = "Taskonomy"
        self.model_name = model_name
        self.device = device

        # Set config path:
        file_path = os.path.abspath(__file__)
        directory_path = os.path.dirname(file_path)
        self.config_path = os.path.join(directory_path, "configs/taskonomy.json")


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
            return Taskonomy.clean_extracted_features
        elif data_type == 'video':
            return Taskonomy.clean_extracted_features
        else:
            raise ValueError(f"Unsupported data type for {self.netset_name}: {data_type}")

    def get_model(self, pretrained):

        # Load attributes from the json
        model_attributes = load_from_json(self.config_path, self.model_name)

        # Set the layers and model function from the attributes
        self.layers = model_attributes["nodes"]
        self.weights = model_attributes["weights"]

        # Inititate the model
        self.loaded_model = model_attributes["model_function"](eval_only=True)

        # Model to device
        self.loaded_model.to(self.device)

        self.loaded_model.eval()

        if pretrained:
            checkpoint = torch.utils.model_zoo.load_url(self.weights) # Load weights
            self.loaded_model.load_state_dict(checkpoint['state_dict'])
        else:
            self.loaded_model.apply(self.randomize_weights)

        return self.loaded_model



    def image_preprocessing(self, image, model_name, device):

        # Colorization needs a different preprocessing to be 1D
        if model_name == 'colorization':
            # Lightness channel of Lab (0-100), rescaled to [-1, 1]
            image = trn.Resize((256, 256))(image.convert('RGB'))
            lightness = rgb2lab(np.asarray(image) / 255.0)[:, :, 0]
            image_tensor = torch.tensor(lightness / 50 - 1, dtype=torch.float32)[None, None]
        else:
            transforms = trn.Compose([
                trn.Resize((256, 256)),
                trn.ToTensor(),
            ])
            # Taskonomy models expect values in [-1, 1]
            image_tensor = transforms(image).unsqueeze(0) * 2 - 1

        # Send to CUDA if needed
        if device == 'cuda':
            image_tensor = image_tensor.cuda()
        return image_tensor


    def video_preprocessing(self, frame, model_name, device):
        return self.image_preprocessing(Image.fromarray(frame), model_name, device)

    def clean_extracted_features(self, features):
        return features
    

            
    
    def extraction_function(self, data, layers_to_extract=None):

        layers = self.select_model_layers(layers_to_extract, self.layers, self.loaded_model)

        # Create a extractor instance
        extractor = create_extractor(self.loaded_model, layers)

        # Extract actual features
        _, features = extractor(data)

        return features


         
