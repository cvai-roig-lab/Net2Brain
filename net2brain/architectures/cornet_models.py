import cv2
from torch.autograd import Variable as V
from torchvision import transforms as trn
from PIL import Image
import torch
#import cornet

from net2brain.architectures.implemented_models.cornet_z import cornet_z
from net2brain.architectures.implemented_models.cornet_rt import cornet_rt
from net2brain.architectures.implemented_models.cornet_s import cornet_s

MODELS = {"cornet_z": cornet_z,
          "cornet_rt": cornet_rt,
          "cornet_s": cornet_s}

MODEL_NODES = {"cornet_z": ['module.V1', 'module.V2', 'module.V4', 'module.IT'],
               "cornet_rt": ['module.V1', 'module.V2', 'module.V4', 'module.IT'],
               "cornet_s": ['module.V1', 'module.V2', 'module.V4', 'module.IT']}

MODEL_WEIGHTS = {"cornet_z": "https://s3.amazonaws.com/cornet-models/cornet_z-5c427c9c.pth",
                 "cornet_rt": "https://s3.amazonaws.com/cornet-models/cornet_rt-933c001c.pth",
                 "cornet_s": "https://s3.amazonaws.com/cornet-models/cornet_s-1d3f7974.pth"}


def preprocess(image, model_name):
    """Preprocesses image according to the networks needs

    Args:
        image (str/path): path to image
        model_name (str): name of the model (sometimes needes to differenciate between model settings)

    Returns:
        PIL-Image: Preprocesses PIL Image
    """

    # Get image
    transforms = trn.Compose([
        trn.Resize((224, 224)),  # resize to 224 x 224 pixels
        trn.ToTensor(),  # transform to tensor
        # normalize according to ImageNet
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = Image.open(image).convert('RGB')
    img = V(transforms(img).unsqueeze(0))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == torch.device('cuda'):  # send to cuda
        img = image.cuda()

    return img


def preprocess_frame(frame, model_name):
    """Preprocesses image according to the networks needs

    Args:
        frame (numpy array): array of frame
        model_name (str): name of the model (sometimes needes to differenciate between model settings)

    Returns:
        PIL-Image: Preprocesses PIL Image
    """
    
    transforms = trn.Compose([
        trn.Resize((224, 224)),  # resize to 224 x 224 pixels
        trn.ToTensor(),  # transform to tensor
        # normalize according to ImageNet
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame)
    
    pil_image = V(transforms(pil_image).unsqueeze(0))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == torch.device('cuda'):  # send to cuda
            pil_image = pil_image.cuda()
            
    return pil_image
