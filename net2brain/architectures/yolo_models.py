import warnings
from PIL import Image
from torchvision import transforms as trn
from .netsetbase import NetSetBase, create_extractor
from .shared_functions import load_from_json
import os

class Yolo(NetSetBase):

    def __init__(self, model_name, device):
        self.supported_data_types = ['image', 'video']
        self.netset_name = "Yolo"
        self.model_name = model_name
        self.device = device

        # Set config path:
        file_path = os.path.abspath(__file__)
        directory_path = os.path.dirname(file_path)
        self.config_path = os.path.join(directory_path, "configs/yolo.json")

    def get_preprocessing_function(self, data_type):
        if data_type == 'image':
            return self.image_preprocessing
        elif data_type == 'video':
            warnings.warn("Models only support image-data. Will average video frames")
            return self.video_preprocessing
        else:
            raise ValueError(f"Unsupported data type for {self.netset_name}: {data_type}")
        

    def image_preprocessing(self, image, model_name, device):
        # YOLOv5 expects values in [0, 1] without normalization
        if not model_name.endswith("6"):
            transforms = trn.Compose([trn.Resize((256, 256)), trn.CenterCrop(224), trn.ToTensor()])
        else:
            # P6 models need input sizes divisible by 64: 256 crop with the same crop ratio as 256 -> 224
            transforms = trn.Compose([trn.Resize((292, 292)), trn.CenterCrop(256), trn.ToTensor()])
        img_tensor = transforms(image).unsqueeze(0)
        return img_tensor.cuda() if device == 'cuda' else img_tensor

    def video_preprocessing(self, frame, model_name, device):
        return self.image_preprocessing(Image.fromarray(frame), model_name, device)

    def get_feature_cleaner(self, data_type):
        if data_type == 'image':
            return Yolo.clean_extracted_features
        elif data_type == 'video':
            return Yolo.clean_extracted_features
        else:
            raise ValueError(f"Unsupported data type for {self.netset_name}: {data_type}")
        

    def get_model(self, pretrained):

        # Load attributes from the json
        model_attributes = load_from_json(self.config_path, self.model_name)

        # Set the layers and model function from the attributes
        self.layers = model_attributes["nodes"]
        self.loaded_model = model_attributes["model_function"]('ultralytics/yolov5', 
                                                               self.model_name, 
                                                               pretrained=pretrained)
        first = next((name for name, _ in self.loaded_model.named_modules() if name.endswith("model.0")), None)
        if first is None:
            raise ValueError(f"Could not find the layers of {self.model_name} (no module ending in 'model.0').")
        self.layers = [first[:-1] + layer.rsplit(".", 1)[-1] for layer in self.layers]

        # Model to device
        self.loaded_model.to(self.device)

        # Randomize weights
        if not pretrained:
            self.loaded_model.apply(self.randomize_weights)

        # Put in eval mode
        self.loaded_model.eval()

        return self.loaded_model


    def clean_extracted_features(self, features):
        cleaned_features = {}
        for layer, output in features.items():
            # Check if the output is a tuple
            if isinstance(output, tuple):
                # Extract the tensor from the tuple
                tensor = output[0]
            else:
                tensor = output
            cleaned_features[layer] = tensor
        return cleaned_features
    
    def extraction_function(self, data, layers_to_extract=None):

        layers = self.select_model_layers(layers_to_extract, self.layers, self.loaded_model)

        # Create a extractor instance
        extractor = create_extractor(self.loaded_model, layers)

        # Extract actual features
        _, features = extractor(data)

        return features

         
