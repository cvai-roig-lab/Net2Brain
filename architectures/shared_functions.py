from torch.autograd import Variable as V
from torchvision import transforms as trn
import cv2
import torch
import torch.nn as nn
from PIL import Image
import warnings
import json

def get_function_from_module(function_string):
    module_name, function_name = function_string.rsplit('.', 1)
    
    # Handle hierarchical module names
    segments = module_name.split('.')

    module = __import__(segments[0])
    for segment in segments[1:]:
        module = getattr(module, segment)
    
    return getattr(module, function_name)


def load_from_json(config_path, model_name):
    # Load the JSON file
    with open(config_path, 'r') as file:
        data = json.load(file)

    # Check if model_name exists in the data
    if model_name not in data:
        raise ValueError(f"{model_name} not found in the configuration file.")

    # Retrieve the attributes for the given model_name
    model_entry = data[model_name]

    # If layers are empty, set them to None
    if not model_entry.get("nodes"):
        warnings.warn("There are no layers preselected, will chose all layers")
        model_entry["nodes"] = None

    # Convert model string to function
    model_string = model_entry.get("model")
    if model_string:
        try:
            model_function = get_function_from_module(model_string)
            model_entry["model_function"] = model_function
        except AttributeError:
            raise ValueError(f"{model_string} is not a valid function name.")
    else:
        raise ValueError(f"Data for {model_name} is incomplete in the configuration file.")

    return model_entry





# Preprocessing

def imagenet_preprocess(image, model_name, device):
    """Preprocesses image provided with path.

    Args:
        image (str/path): path to image
        model_name (str): name of the model

    Returns:
        PIL-Image: Preprocesses PIL Image
    """
    transforms = trn.Compose([
        trn.Resize((224, 224)),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = V(transforms(image).unsqueeze(0))

    if device == 'cuda':
        img = img.cuda()

    return img


def imagenet_preprocess_frames(frame, model_name, device):
    """Preprocesses image provided in frame.

    Args:
        frame (numpy array): array of frame
        model_name (str): name of the model

    Returns:
        PIL-Image: Preprocesses PIL Image
    """
    centre_crop = trn.Compose([
        trn.Resize((224, 224)),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    pil_image = V(centre_crop(frame).unsqueeze(0))

    if device == 'cuda':  # send to cuda
        pil_image = pil_image.cuda()

    return pil_image



def randomize_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)



def torch_clean(features):
    """Cleanup function after feature extraction: 
    This one contains subdictionaries which need to be eliminated.

    Args:
        features (dict:tensors): dictionary of tensors

    Returns:
        (dict:tensors): dictionary of tensors
    """
    new_features = {}
    for key, value in features.items():
        try:
            new_features[key] = value["out"].data.cpu()
        except:
            new_features[key] = value.data.cpu()
    return new_features




def extract_features_tx(image):
    """Extract features with torch extractor.

    Parameters
    ----------
    image : Torch Tensor
        Preprocessed image.

    Returns
    -------
    dict of Torch Tensors
        Features by layer.
    """
    extractor = tx.Extractor(self.model, self.layers_to_extract)
    _, features = extractor(image)
    features = self._features_cleaner(features)
    return features

def _extract_features_tx_clip(self, image):
    """Extract CLIP features with torch extractor.

    Parameters
    ----------
    image : Torch Tensor
        Preprocessed image.

    Returns
    -------
    dict of Torch Tensors
        Features by layer.
    """
    extractor = tx.Extractor(self.model, self.layers_to_extract)
    image_data = image[0]
    tokenized_data = image[1]
    _, features = extractor(image_data, tokenized_data)
    features = self._features_cleaner(features)
    return features

def _extract_features_timm(self, image):
    """Extract features with timm.

    Parameters
    ----------
    image : Torch Tensor
        Preprocessed image.

    Returns
    -------
    dict of Torch Tensors
        Features by layer.
    """
    features = self.model(image)
    # Convert the features into a dict because timm extractor returns a 
    # list of tensors
    converted_features = {}
    for counter, feature in enumerate(features):
        converted_features[f"feature {str(counter+1)}"] = feature.data.cpu()
    features = self._features_cleaner(converted_features)
    return converted_features

def _no_clean(self, features):
    """Cleanup after feature extraction: This one requires no cleanup.
    Just put it on cpu in case it isn't yet!

    Args:
        features (dict:tensors): dictionary of tensors

    Returns:
        (dict:tensors): dictionary of tensors
    """
    return {key: value.data.cpu() for key, value in features.items()}

def _torch_clean(self, features):
    """Cleanup function after feature extraction: 
    This one contains subdictionaries which need to be eliminated.

    Args:
        features (dict:tensors): dictionary of tensors

    Returns:
        (dict:tensors): dictionary of tensors
    """
    new_features = {}
    for key, value in features.items():
        try:
            new_features[key] = value["out"].data.cpu()
        except:
            new_features[key] = value.data.cpu()
    return new_features