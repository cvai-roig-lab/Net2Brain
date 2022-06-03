import torch
import cv2
from torch.autograd import Variable as V
from torchvision import transforms as trn
from PIL import Image

MODELS = {'yolov5l': torch.hub.load,
          'yolov5l6': torch.hub.load,
          'yolov5m': torch.hub.load,
          'yolov5m6': torch.hub.load,
          'yolov5n': torch.hub.load,
          'yolov5n6': torch.hub.load,
          'yolov5s': torch.hub.load,
          'yolov5s6': torch.hub.load,
          'yolov5x': torch.hub.load,
          'yolov5x6': torch.hub.load}


MODEL_NODES = {'yolov5l': ['model.model.0', 'model.model.1', 'model.model.2',
                           'model.model.3', 'model.model.4', 'model.model.5',
                           'model.model.6', 'model.model.7', 'model.model.8',
                           'model.model.9', 'model.model.10', 'model.model.11', 'model.model.12', 'model.model.13',
                           'model.model.14', 'model.model.15', 'model.model.16', 'model.model.17', 'model.model.18',
                           'model.model.19', 'model.model.20', 'model.model.21', 'model.model.22', 'model.model.23', 'model.model.24'],
               'yolov5l6': ['model.model.0', 'model.model.1', 'model.model.2',
                            'model.model.3', 'model.model.4', 'model.model.5',
                            'model.model.6', 'model.model.7', 'model.model.8',
                            'model.model.9', 'model.model.10', 'model.model.11', 'model.model.12', 'model.model.13',
                            'model.model.14', 'model.model.15', 'model.model.16', 'model.model.17', 'model.model.18',
                            'model.model.19', 'model.model.20', 'model.model.21', 'model.model.22', 'model.model.23', 'model.model.24'],
               'yolov5m': ['model.model.0', 'model.model.1', 'model.model.2',
                           'model.model.3', 'model.model.4', 'model.model.5',
                           'model.model.6', 'model.model.7', 'model.model.8',
                           'model.model.9', 'model.model.10', 'model.model.11', 'model.model.12', 'model.model.13',
                           'model.model.14', 'model.model.15', 'model.model.16', 'model.model.17', 'model.model.18',
                           'model.model.19', 'model.model.20', 'model.model.21', 'model.model.22', 'model.model.23', 'model.model.24'],
               'yolov5m6': ['model.model.0', 'model.model.1', 'model.model.2',
                            'model.model.3', 'model.model.4', 'model.model.5',
                            'model.model.6', 'model.model.7', 'model.model.8',
                            'model.model.9', 'model.model.10', 'model.model.11', 'model.model.12', 'model.model.13',
                            'model.model.14', 'model.model.15', 'model.model.16', 'model.model.17', 'model.model.18',
                            'model.model.19', 'model.model.20', 'model.model.21', 'model.model.22', 'model.model.23', 'model.model.24', 'model.model.25',
                            'model.model.26', 'model.model.27', 'model.model.28', 'model.model.29', 'model.model.30', 'model.model.31', 'model.model.32',
                            'model.model.33'],
               'yolov5n': ['model.model.0', 'model.model.1', 'model.model.2',
                           'model.model.3', 'model.model.4', 'model.model.5',
                           'model.model.6', 'model.model.7', 'model.model.8',
                           'model.model.9', 'model.model.10', 'model.model.11', 'model.model.12', 'model.model.13',
                           'model.model.14', 'model.model.15', 'model.model.16', 'model.model.17', 'model.model.18',
                           'model.model.19', 'model.model.20', 'model.model.21', 'model.model.22', 'model.model.23', 'model.model.24'],
               'yolov5n6': ['model.model.0', 'model.model.1', 'model.model.2',
                            'model.model.3', 'model.model.4', 'model.model.5',
                            'model.model.6', 'model.model.7', 'model.model.8',
                            'model.model.9', 'model.model.10', 'model.model.11', 'model.model.12', 'model.model.13',
                            'model.model.14', 'model.model.15', 'model.model.16', 'model.model.17', 'model.model.18',
                            'model.model.19', 'model.model.20', 'model.model.21', 'model.model.22', 'model.model.23', 'model.model.24', 'model.model.25',
                            'model.model.26', 'model.model.27', 'model.model.28', 'model.model.29', 'model.model.30', 'model.model.31', 'model.model.32',
                            'model.model.33'],
               'yolov5s': ['model.model.0', 'model.model.1', 'model.model.2',
                           'model.model.3', 'model.model.4', 'model.model.5',
                           'model.model.6', 'model.model.7', 'model.model.8',
                           'model.model.9', 'model.model.10', 'model.model.11', 'model.model.12', 'model.model.13',
                           'model.model.14', 'model.model.15', 'model.model.16', 'model.model.17', 'model.model.18',
                           'model.model.19', 'model.model.20', 'model.model.21', 'model.model.22', 'model.model.23', 'model.model.24'],
               'yolov5s6': ['model.model.0', 'model.model.1', 'model.model.2',
                            'model.model.3', 'model.model.4', 'model.model.5',
                            'model.model.6', 'model.model.7', 'model.model.8',
                            'model.model.9', 'model.model.10', 'model.model.11', 'model.model.12', 'model.model.13',
                            'model.model.14', 'model.model.15', 'model.model.16', 'model.model.17', 'model.model.18',
                            'model.model.19', 'model.model.20', 'model.model.21', 'model.model.22', 'model.model.23', 'model.model.24', 'model.model.25',
                            'model.model.26', 'model.model.27', 'model.model.28', 'model.model.29', 'model.model.30', 'model.model.31', 'model.model.32',
                            'model.model.33'],
               'yolov5x': ['model.model.0', 'model.model.1', 'model.model.2',
                           'model.model.3', 'model.model.4', 'model.model.5',
                           'model.model.6', 'model.model.7', 'model.model.8',
                           'model.model.9', 'model.model.10', 'model.model.11', 'model.model.12', 'model.model.13',
                           'model.model.14', 'model.model.15', 'model.model.16', 'model.model.17', 'model.model.18',
                           'model.model.19', 'model.model.20', 'model.model.21', 'model.model.22', 'model.model.23', 'model.model.24'],
               'yolov5x6': ['model.model.0', 'model.model.1', 'model.model.2',
                            'model.model.3', 'model.model.4', 'model.model.5',
                            'model.model.6', 'model.model.7', 'model.model.8',
                            'model.model.9', 'model.model.10', 'model.model.11', 'model.model.12', 'model.model.13',
                            'model.model.14', 'model.model.15', 'model.model.16', 'model.model.17', 'model.model.18',
                            'model.model.19', 'model.model.20', 'model.model.21', 'model.model.22', 'model.model.23', 'model.model.24', 'model.model.25',
                            'model.model.26', 'model.model.27', 'model.model.28', 'model.model.29', 'model.model.30', 'model.model.31', 'model.model.32',
                            'model.model.33']}


def preprocess(image, model_name):
    """Preprocesses image according to the networks needs

    Args:
        image (str/path): path to image
        model_name (str): name of the model (sometimes needes to differenciate between model settings)

    Returns:
        PIL-Image: Preprocesses PIL Image
    """

    centre_crop = trn.Compose([
        trn.Resize((224, 224)),  # resize to 224 x 224 pixels
        trn.ToTensor(),  # transform to tensor
        # normalize according to ImageNet
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(image)
    
    image = V(centre_crop(image).unsqueeze(0))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == torch.device('cuda'):  # send to cuda
        image = image.cuda()

    return image


def preprocess_frame(frame, model_name):
    """Preprocesses image according to the networks needs

    Args:
        frame (numpy array): array of frame
        model_name (str): name of the model (sometimes needes to differenciate between model settings)

    Returns:
        PIL-Image: Preprocesses PIL Image
    """
    
    centre_crop = trn.Compose([
        trn.Resize((224, 224)),  # resize to 224 x 224 pixels
        trn.ToTensor(),  # transform to tensor
        # normalize according to ImageNet
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame)
    
    pil_image = V(centre_crop(pil_image).unsqueeze(0))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == torch.device('cuda'):  # send to cuda
            pil_image = pil_image.cuda()
            
    return pil_image

