import cv2
from torch.autograd import Variable as V
from torchvision import transforms as trn
from PIL import Image
import torch
import cornet


MODELS = {"cornet_z": cornet.CORnet_Z,
          "cornet_rt": cornet.CORnet_RT,
          "cornet_s": cornet.CORnet_S}

MODEL_NODES = {"cornet_z": ['module.V1', 'module.V2', 'module.V3', 'module.V4', 'module.IT'],
               "cornet_rt": ['module.V1', 'module.V2', 'module.V3', 'module.V4', 'module.IT'],
               "cornet_s": ['module.V1', 'module.V2', 'module.V3', 'module.V4', 'module.IT']}

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
