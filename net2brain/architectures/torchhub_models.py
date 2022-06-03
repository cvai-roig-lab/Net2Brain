import torch
import cv2
from torch.autograd import Variable as V
from torchvision import transforms as trn
from PIL import Image
import torch

MODELS = {'deeplabv3_mobilenet_v3_large': torch.hub.load,
          'deeplabv3_resnet101': torch.hub.load,
          'deeplabv3_resnet50': torch.hub.load,
          'fcn_resnet101': torch.hub.load,
          'fcn_resnet50': torch.hub.load,
          'lraspp_mobilenet_v3_large': torch.hub.load}


MODEL_NODES = {'deeplabv3_mobilenet_v3_large': ['backbone.1', 'backbone.2', 'backbone.3', 'backbone.4', 'backbone.5', 'backbone.6', 'backbone.7'
                                                'backbone.8', 'backbone.9', 'backbone.10', 'backbone.11', 'backbone.12', 'backbone.13', 'backbone.14', 'backbone.15'
                                                'backbone.16', 'classifier', 'classifier.1', 'classifier.2', 'classifier.3', 'classifier.4'],
               'deeplabv3_resnet101': ['backbone.layer1', 'backbone.layer2', 'backbone.layer3', 'backbone.layer4',  'classifier', 'classifier.1',
                                       'classifier.2', 'classifier.3', 'classifier.4'],
               'deeplabv3_resnet50': ['backbone.layer1', 'backbone.layer2', 'backbone.layer3', 'backbone.layer4',  'classifier', 'classifier.1',
                                       'classifier.2', 'classifier.3', 'classifier.4'],
               'fcn_resnet101': ['backbone.layer1', 'backbone.layer2', 'backbone.layer3', 'backbone.layer4',  'classifier', 'classifier.1',
                                 'classifier.2', 'classifier.3', 'classifier.4'],
               'fcn_resnet50': ['backbone.layer1', 'backbone.layer2', 'backbone.layer3', 'backbone.layer4',  'classifier', 'classifier.1',
                                 'classifier.2', 'classifier.3', 'classifier.4'],
               'lraspp_mobilenet_v3_large': [ 'backbone.1', 'backbone.2', 'backbone.3', 'backbone.4', 'backbone.5', 'backbone.6', 'backbone.7'
                                                'backbone.8', 'backbone.9', 'backbone.10', 'backbone.11', 'backbone.12', 'backbone.13', 'backbone.14', 'backbone.15'
                                                'backbone.16',]}


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
