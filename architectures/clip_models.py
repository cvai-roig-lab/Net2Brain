import warnings
from .netsetbase import NetSetBase
from .shared_functions import imagenet_preprocess, imagenet_preprocess_frames, load_from_json
import torchextractor as tx
import clip
import torch


class Clip(NetSetBase):

    def __init__(self, model_name, device):
        self.supported_data_types = ['image', 'video']
        self.netset_name = "Clip"
        self.model_name = model_name
        self.device = device


    def get_preprocessing_function(self, data_type):
        if data_type == 'image':
            return Clip.image_preprocessing
        elif data_type == 'video':
            warnings.warn("Models only support image-data. Will average video frames")
            return Clip.video_preprocessing
        else:
            raise ValueError(f"Unsupported data type for {self.netset_name}: {data_type}")
        

    def get_feature_cleaner(self, data_type):
        if data_type == 'image':
            return Clip.clean_extracted_features
        elif data_type == 'video':
            return Clip.clean_extracted_features
        else:
            raise ValueError(f"Unsupported data type for {self.netset_name}: {data_type}")
        

    def get_model(self, pretrained):

        # Change model name
        self.model_name = self.model_name.replace("_-_", "/")

        # Set configuration path 
        config_path = "architectures/configs/clip.json"

        # Load attributes from the json
        model_attributes = load_from_json(config_path, self.model_name)

        # Set the layers and model function from the attributes
        self.layers = model_attributes["nodes"]
        self.loaded_model = model_attributes["model_function"](self.model_name, device=self.device)[0]

        # Model to device
        self.loaded_model.to(self.device)

        # Randomize weights
        if not pretrained:
            self.loaded_model.apply(self.randomize_weights)
            
        # Put in eval mode
        self.loaded_model.eval()


    def image_preprocessing(self, image, model_name, device):
        image = imagenet_preprocess(image, model_name, device)
        text = torch.cat([clip.tokenize("a photo of a word")])

        # Send to device
        if device == 'cuda':
            image = image.cuda()
            text = text.cuda()

        return [image, text]

    def video_preprocessing(self, frame, model_name, device):
        image = imagenet_preprocess_frames(frame, model_name, device)
        text = torch.cat([clip.tokenize("a photo of a word")])

        # Send to device
        if device == 'cuda':
            image = image.cuda()
            text = text.cuda()

        return [image, text]


    def clean_extracted_features(self, features):
        return features
    
        
    
    def extraction_function(self, data, layers_to_extract=None):

        self.layers = self.select_model_layers(layers_to_extract, self.layers, self.loaded_model)

        # Create a extractor instance
        self.extractor_model = tx.Extractor(self.loaded_model, self.layers)

        # Seperate image and text data
        image_data = data[0]
        tokenized_text = data[1]

        # Extract actual features
        _, features = self.extractor_model(image_data, tokenized_text)

        return features

         
