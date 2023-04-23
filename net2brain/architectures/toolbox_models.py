import torchvision.models as models
import cv2
from torch.autograd import Variable as V
from torchvision import transforms as trn
from PIL import Image
import torch


from net2brain.architectures.implemented_models.places365_net import get_resnet50_places365
from net2brain.architectures.implemented_models.semseg_models import get_semseg_model

MODELS = {"Places365": get_resnet50_places365,
          "SemSeg": get_semseg_model}

MODEL_NODES = {"Places365": ["model.4", "model.5", "model.6", "model.7", "model.9"],
               "SemSeg": ['decoder.ppm_conv.0.2','decoder.ppm_conv.2.2','decoder.ppm_conv.3.2','decoder.ppm_last_conv.2','decoder.fpn_in.2.2','decoder.fpn_out.2.0.2','decoder.conv_last.1']}




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

    if device == 'cuda':  # send to cudaa
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
    
    centre_crop = trn.Compose([
        trn.Resize((224, 224)),  # resize to 224 x 224 pixels
        trn.ToTensor(),  # transform to tensor
        # normalize according to ImageNet
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame)
    
    pil_image = V(centre_crop(pil_image).unsqueeze(0))

    if device == 'cuda':  # send to cudaa
            pil_image = pil_image.cuda()
            
    return pil_image
