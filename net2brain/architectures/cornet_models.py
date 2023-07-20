import cv2
from torch.autograd import Variable as V
from torchvision import transforms as trn
from PIL import Image
#import cornet

from net2brain.architectures.implemented_models.cornet_z import cornet_z
from net2brain.architectures.implemented_models.cornet_rt import cornet_rt
from net2brain.architectures.implemented_models.cornet_s import cornet_s

MODELS = {"cornet_z": cornet_z,
          "cornet_rt": cornet_rt,
          "cornet_s": cornet_s}

MODEL_NODES = {"cornet_z": ['V1', 'V2', 'V4', 'IT'],
               "cornet_rt": ['V1', 'V2', 'V4', 'IT'],
               "cornet_s": ['V1', 'V2', 'V4', 'IT']}
               

def preprocess(image, model_name, device):
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
    
    if device == 'cuda':  # send to cuda
        img = img.cuda()

    return img


def preprocess_frame(frame, model_name, device):
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
    
    if device == 'cuda':  # send to cuda
            pil_image = pil_image.cuda()
            
    return pil_image
