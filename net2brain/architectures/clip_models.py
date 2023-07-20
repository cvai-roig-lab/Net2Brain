import cv2
from PIL import Image

import clip
import torch
from torch.autograd import Variable as V
from torchvision import transforms as trn


MODELS = {'RN50': clip.load,
          'RN101': clip.load,
          'ViT-B_-_32': clip.load,
          'ViT-B_-_16': clip.load,
          'ViT-L_-_14': clip.load}

MODEL_NODES = {
    'RN50': [f'visual.layer{i}' for i in range(1, 5)],
    'RN101': [f'visual.layer{i}' for i in range(1, 5)],
    'ViT-B_-_32': [f'visual.transformer.resblocks.{i}' for i in range(12)],
    'ViT-B_-_16': [f'visual.transformer.resblocks.{i}' for i in range(12)],
    'ViT-L_-_14': [f'visual.transformer.resblocks.{i}' for i in range(24)]
}


def preprocess(image, model_name, device):
    """Preprocesses image.

    Args:
        image (str/path): path to image
        model_name (str): name of the model

    Returns:
        PIL-Image: Preprocesses PIL Image
    """

    # Transform image
    transforms = trn.Compose([
        trn.Resize((224, 224)),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = Image.open(image).convert('RGB')
    img = V(transforms(img).unsqueeze(0))

    # Create prompt and tokenize
    txt = torch.cat([clip.tokenize(f"a photo of a {c}") for c in ["word"]])

    # Send to device
    if device == 'cuda':
        img = img.cuda()
        txt = txt.cuda()

    return [img, txt]


def preprocess_frame(frame, model_name, device):
    """Preprocesses image according to the networks needs

    Args:
        frame (numpy array): array of frame
        model_name (str): name of the model

    Returns:
        PIL-Image: Preprocesses PIL Image
    """
    transforms = trn.Compose([
        trn.Resize((224, 224)),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    img = V(transforms(img).unsqueeze(0))

    # Create prompt and tokenize
    txt = torch.cat([clip.tokenize(f"a photo of a {c}") for c in ["word"]])

    # Send to device
    if device == 'cuda':
        img = img.cuda()
        txt = txt.cuda()

    return [img, txt]
