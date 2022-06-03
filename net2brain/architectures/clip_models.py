
import cv2
from torch.autograd import Variable as V
from torchvision import transforms as trn
from PIL import Image
import clip
import torch

MODELS = {'RN50': clip.load,
          'RN101': clip.load,
          'ViT-B_-_32': clip.load,
          'ViT-B_-_16': clip.load,
          'ViT-L_-_14': clip.load}


MODEL_NODES = {'RN50': ['visual.layer1', 'visual.layer2', 'visual.layer3', 'visual.layer4'],
               'RN101': [ 'visual.layer1', 'visual.layer2', 'visual.layer3', 'visual.layer4'],
               'ViT-B_-_32': ['visual.transformer.resblocks.0',
                        'visual.transformer.resblocks.1', 'visual.transformer.resblocks.2',
                        'visual.transformer.resblocks.3', 'visual.transformer.resblocks.4', 'visual.transformer.resblocks.5',
                        'visual.transformer.resblocks.6', 'visual.transformer.resblocks.7', 'visual.transformer.resblocks.8', 
                        'visual.transformer.resblocks.9', 'visual.transformer.resblocks.10', 'visual.transformer.resblocks.11'],
               'ViT-B_-_16': ['visual.transformer.resblocks.0',
                            'visual.transformer.resblocks.1', 'visual.transformer.resblocks.2',
                            'visual.transformer.resblocks.3', 'visual.transformer.resblocks.4', 'visual.transformer.resblocks.5',
                            'visual.transformer.resblocks.6', 'visual.transformer.resblocks.7', 'visual.transformer.resblocks.8',
                            'visual.transformer.resblocks.9', 'visual.transformer.resblocks.10', 'visual.transformer.resblocks.11'],
               'ViT-L_-_14': ['visual.transformer.resblocks.0',
                            'visual.transformer.resblocks.1', 'visual.transformer.resblocks.2',
                            'visual.transformer.resblocks.3', 'visual.transformer.resblocks.4', 'visual.transformer.resblocks.5',
                            'visual.transformer.resblocks.6', 'visual.transformer.resblocks.7', 'visual.transformer.resblocks.8',
                            'visual.transformer.resblocks.9', 'visual.transformer.resblocks.10', 'visual.transformer.resblocks.11',
                            'visual.transformer.resblocks.12', 'visual.transformer.resblocks.13', 'visual.transformer.resblocks.14', 
                            'visual.transformer.resblocks.15', 'visual.transformer.resblocks.16', 'visual.transformer.resblocks.17', 
                            'visual.transformer.resblocks.18', 'visual.transformer.resblocks.19', 'visual.transformer.resblocks.20', 
                            'visual.transformer.resblocks.21', 'visual.transformer.resblocks.22', 'visual.transformer.resblocks.23']}


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
        
    # For clip we need to add text embedding. Since we dont extract word embedding, that can be random
    tokenized_text = torch.cat([clip.tokenize(f"a photo of a {c}") for c in ["word"]]).to("cpu") 
    
    image = V(centre_crop(image).unsqueeze(0))
    
    # Add to Cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == torch.device('cuda'):  # send to cuda
        image = image.cuda()
        tokenized_text = tokenized_text.cuda()

    return [image, tokenized_text]


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
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # For clip we need to add text embedding. Since we dont extract word embedding, that can be random
    tokenized_text = torch.cat([clip.tokenize(f"a photo of a {c}") for c in ["word"]]).to("cpu")
    
    pil_image = V(centre_crop(pil_image).unsqueeze(0))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == torch.device('cuda'):  # send to cuda
        pil_image = pil_image.cuda()
        tokenized_text = tokenized_text.cuda()
        
    return [pil_image, tokenized_text]
