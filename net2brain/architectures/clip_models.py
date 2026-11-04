import warnings
from .netsetbase import NetSetBase, create_extractor
from .shared_functions import load_from_json
import clip
import torch
import os
import numpy as np
from PIL import Image


def random_clip_like(model):
    """A CLIP model with the same architecture as `model`, but randomly initialised."""
    visual, is_vit = model.visual, hasattr(model.visual, "proj")
    return type(model)(
        embed_dim=model.text_projection.shape[1],
        image_resolution=visual.input_resolution,
        vision_layers=visual.transformer.layers if is_vit else tuple(len(getattr(visual, f"layer{i}")) for i in range(1, 5)),
        vision_width=visual.conv1.out_channels if is_vit else visual.layer1[0].conv1.out_channels,
        vision_patch_size=visual.conv1.kernel_size[0] if is_vit else None,
        context_length=model.context_length, vocab_size=model.vocab_size,
        transformer_width=model.transformer.width, transformer_heads=model.transformer.width // 64,
        transformer_layers=model.transformer.layers)


class Clip(NetSetBase):

    def __init__(self, model_name, device):
        self.supported_data_types = ['image', 'video', 'multimodal', 'text']
        self.netset_name = "Clip"
        self.model_name = model_name
        self.device = device

        # Set config path:
        file_path = os.path.abspath(__file__)
        directory_path = os.path.dirname(file_path)
        self.config_path = os.path.join(directory_path, "configs/clip.json")



    def get_preprocessing_function(self, data_type):
        if data_type == 'image':
            warnings.warn("This is a multimodal model. With unimodal input the text-transformer layers will have the same activations throughout all images.")
            return self.image_preprocessing
        elif data_type == 'video':
            warnings.warn("This is a multimodal model. With unimodal input the text-transformer layers will have the same activations throughout all images.")
            warnings.warn("Models only support image-data. Will average video frames")
            return self.video_preprocessing
        elif data_type == 'text':
            warnings.warn("This is a multimodal model. With unimodal input the vision-transformer layers will have the same activations throughout all text files.")
            return self.text_preprocessing
        elif data_type == "multimodal":
            return self.multimodal_preprocessing
        else:
            raise ValueError(f"Unsupported data type for {self.netset_name}: {data_type}")
        

    def get_feature_cleaner(self, data_type):
        if data_type == 'image':
            return Clip.clean_extracted_features
        elif data_type == 'video':
            return Clip.clean_extracted_features
        elif data_type == 'multimodal':
            return Clip.clean_extracted_features
        elif data_type == 'text':
            return Clip.clean_extracted_features
        else:
            raise ValueError(f"Unsupported data type for {self.netset_name}: {data_type}")
        

    def get_model(self, pretrained):

        # Change model name
        self.model_name = self.model_name.replace("_-_", "/")

        # Load attributes from the json
        model_attributes = load_from_json(self.config_path, self.model_name)

        # Set the layers and model function from the attributes
        self.layers = model_attributes["nodes"]
        self.loaded_model = model_attributes["model_function"](self.model_name, device=self.device)[0]
        if not pretrained:  # clip.load always loads pretrained weights
            self.loaded_model = random_clip_like(self.loaded_model)

        # Model to device
        self.loaded_model.to(self.device)

        # Randomize weights
        if not pretrained:
            self.loaded_model.apply(self.randomize_weights)
            
        # Put in eval mode
        self.loaded_model.eval()

        return self.loaded_model


    def image_preprocessing(self, image, model_name, device):
        image = super().image_preprocessing(image, model_name, device)
        text = torch.cat([clip.tokenize("a photo of a word")])
        
        # Send to device
        if device == 'cuda':
            image = image.cuda()
            text = text.cuda()

        return [image, text]

    def video_preprocessing(self, frame, model_name, device):
        image = super().video_preprocessing(frame, model_name, device)
        text = torch.cat([clip.tokenize("a photo of a word")])

        # Send to device
        if device == 'cuda':
            image = image.cuda()
            
            text = text.cuda()

        return [image, text]
    
    
    def text_preprocessing(self, text, model_name, device):
        
        # Create random dummy image
        dummy_image = dummy_image_data = np.random.rand(224, 224, 3) * 255 
        dummy_image = Image.fromarray(dummy_image_data.astype('uint8'), 'RGB')
        image = super().image_preprocessing(dummy_image, model_name, device)
        
        # Loading text
        text_raw = super().text_preprocessing(text, model_name, device)
        text = torch.cat([clip.tokenize(text_raw)])
        
        # Send to device
        if device == 'cuda':
            image = image.cuda()
            text = text.cuda()

        return [image, text]
    
    
    def multimodal_preprocessing(self, tuple_path, model_name, device):
        loaded_image, loaded_text = tuple_path
        image = super().image_preprocessing(loaded_image, model_name, device)
        text_raw = super().text_preprocessing(loaded_text, model_name, device)
        text = torch.cat([clip.tokenize(text_raw)])
        
        # Send to device
        if device == 'cuda':
            image = image.cuda()
            text = text.cuda()

        return [image, text]
        


    def clean_extracted_features(self, features):
        return features
    
        
    
    def extraction_function(self, data, layers_to_extract=None):

        layers = self.select_model_layers(layers_to_extract, self.layers, self.loaded_model)

        # Create a extractor instance
        extractor = create_extractor(self.loaded_model, layers)

        # Seperate image and text data
        image_data = data[0]
        tokenized_text = data[1]

        # Extract actual features
        _, features = extractor(image_data, tokenized_text)

        return features

         
