import torch
import cv2
from torch.autograd import Variable as V
from torchvision import transforms as trn
from PIL import Image

MODELS = {'unet': torch.hub.load}

MODEL_NODES = {'unet': ['encoder1', 'encoder2', 'encoder3', 'encoder4', 'decoder4', 'decoder3', 'decoder2', 'decoder1', 'conv']}

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


