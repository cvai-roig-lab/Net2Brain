
import warnings
from .netsetbase import NetSetBase
from torchvision import transforms as trn
from .shared_functions import imagenet_preprocess, imagenet_preprocess_frames, torch_clean, load_from_json
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
import visualpriors
from .implemented_models import taskonomy_model
import torch
import torchextractor as tx

class Taskonomy(NetSetBase):

    def __init__(self, model_name):
        self.supported_data_types = ['image', 'video']
        self.netset_name = "Taskonomy"
        self.model_name = model_name


    def get_preprocessing_function(self, data_type):
        if data_type == 'image':
            return Taskonomy.image_preprocessing
        elif data_type == 'video':
            warnings.warn("Models only support image-data. Will average video frames")
            return Taskonomy.video_preprocessing
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

        # Set configuration path 
        config_path = "architectures/configs/taskonomy.json"

        # Load attributes from the json
        model_attributes = load_from_json(config_path, self.model_name)

        # Set the layers and model function from the attributes
        self.layers = model_attributes["nodes"]
        self.weights = model_attributes["weights"]

        # Inititate the model
        self.loaded_model = model_attributes["model_function"](eval_only=True)

        self.loaded_model.eval()

        if pretrained:
            checkpoint = torch.utils.model_zoo.load_url(self.weights) # Load weights
            self.loaded_model.load_state_dict(checkpoint['state_dict'])



    def image_preprocessing(self, image, model_name, device):

        # Colorization needs a different preprocessing to be 1D
        if model_name == 'colorization':
            image = image.convert('L')
            transforms = trn.Compose([
                trn.Resize((224, 224)),
                trn.ToTensor(),
            ])
            image_tensor = transforms(image).unsqueeze(0)

            # Send to CUDA if needed
            if device == 'cuda':
                image_tensor = image_tensor.cuda()
            return image_tensor
        else:
            return imagenet_preprocess(image, model_name, device)


    def video_preprocessing(self, frame, model_name, device):
        
        # Colorization needs a different preprocessing to be 1D
        if model_name == 'colorization':
            image = image.convert('L')
            transforms = trn.Compose([
                trn.Resize((224, 224)),
                trn.ToTensor(),
            ])
            image_tensor = transforms(image).unsqueeze(0)

            # Send to CUDA if needed
            if device == 'cuda':
                image_tensor = image_tensor.cuda()
            return image_tensor
        else:
            return imagenet_preprocess_frames(frame, model_name, device)

    def clean_extracted_features(self, features):
        return features
    

            
    
    def extraction_function(self, data, layers_to_extract=None):

        self.layers = self.select_model_layers(layers_to_extract, self.layers, self.loaded_model)

        # Create a extractor instance
        self.extractor_model = create_feature_extractor(self.loaded_model, return_nodes=self.layers)

        return self.extractor_model(data)

         
