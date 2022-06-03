import torchvision.models as models
import cv2
from torch.autograd import Variable as V
from torchvision import transforms as trn
from PIL import Image
import torch

MODELS = {"AlexNet": models.alexnet,
          "ResNet18": models.resnet18,
          "ResNet34": models.resnet34,
          "ResNet50": models.resnet50,
          "ResNet101": models.resnet101,
          "ResNet152": models.resnet152,
          "Squeezenet1_0": models.squeezenet1_0,
          "Squeezenet1_1": models.squeezenet1_1,
          "VGG11": models.vgg11,
          "VGG11_bn": models.vgg11_bn,
          "VGG13": models.vgg13,
          "VGG13_bn": models.vgg13_bn,
          "VGG16": models.vgg16,
          "VGG16_bn": models.vgg16_bn,
          "VGG19": models.vgg19,
          "VGG19_bn": models.vgg19_bn,
          "Densenet121": models.densenet121,
          "Densenet161": models.densenet161,
          "Densenet169": models.densenet169,
          "Densenet201": models.densenet201,
          "GoogleNet": models.googlenet,
          "ShuffleNetV2x05": models.shufflenet_v2_x0_5,
          "ShuffleNetV2x10": models.shufflenet_v2_x1_0,
          "mobilenet_v2": models.mobilenet_v2,
          "mobilenet_v3_large": models.mobilenet_v3_large,
          "mobilenet_v3_small": models.mobilenet_v3_small,
          "resnext50_32x4d": models.resnext50_32x4d,
          "resnext101_32x8d": models.resnext101_32x8d,
          "wide_resnet101_2": models.wide_resnet101_2,
          "wide_resnet50_2": models.wide_resnet50_2,
          "mnasnet05": models.mnasnet0_5,
          "mnasnet10": models.mnasnet1_0,
          "efficientnet_b0": models.efficientnet_b0,
          "efficientnet_b1": models.efficientnet_b1,
          "efficientnet_b2": models.efficientnet_b2,
          "efficientnet_b3": models.efficientnet_b3,
          "efficientnet_b4": models.efficientnet_b4,
          "efficientnet_b5": models.efficientnet_b5,
          "efficientnet_b6": models.efficientnet_b6,
          "efficientnet_b7": models.efficientnet_b7,
          "regnet_y_400mf": models.regnet_y_400mf,
          "regnet_y_800mf": models.regnet_y_800mf,
          "regnet_y_1_6gf": models.regnet_y_1_6gf,
          "regnet_y_3_2gf": models.regnet_y_3_2gf,
          "regnet_y_8gf": models.regnet_y_8gf,
          "regnet_y_16gf": models.regnet_y_16gf,
          "regnet_y_32gf": models.regnet_y_32gf,
          "regnet_x_400mf": models.regnet_x_400mf,
          "regnet_x_800mf": models.regnet_x_800mf,
          "regnet_x_1_6gf": models.regnet_x_1_6gf,
          "regnet_x_3_2gf": models.regnet_x_3_2gf,
          "regnet_x_8gf": models.regnet_x_8gf,
          "regnet_x_16gf": models.regnet_x_16gf,
          "regnet_x_32gf": models.regnet_x_32gf}

MODEL_NODES = {"AlexNet": ['features.0', 'features.3', 'features.6',
                           'features.8', 'features.10'],
               "ResNet18": ['layer1', 'layer2', 'layer3', 'layer4'],
               "ResNet34": ['layer1', 'layer2', 'layer3', 'layer4'],
               "ResNet50": ['layer1', 'layer2', 'layer3', 'layer4'],
               "ResNet101": ['layer1', 'layer2', 'layer3', 'layer4'],
               "ResNet152": ['layer1', 'layer2', 'layer3', 'layer4'],
               "Squeezenet1_0": ['features.0', 'features.3', 'features.4', 'features.5', 'features.7', 'features.8', 'features.9', 'features.10', 'features.11', 'features.12'],
               "Squeezenet1_1": ['features.0', 'features.3', 'features.4', 'features.6', 'features.7', 'features.9', 'features.10', 'features.12'],
               "VGG11": ['features.0', 'features.3', 'features.6', 'features.8', 'features.11', 'features.13', 'features.16', 'features.18',],
               "VGG11_bn": ['features.0', 'features.4', 'features.8', 'features.11', 'features.15', 'features.18', 'features.22', 'features.25'],
               "VGG13": ['features.0', 'features.2', 'features.5', 'features.7', 'features.10', 'features.12', 'features.15', 'features.17', 'features.20', 'features.22'],
               "VGG13_bn": ['features.0', 'features.3', 'features.7', 'features.10', 'features.14', 'features.17', 'features.21', 'features.24', 'features.28', 'features.31'],
               "VGG16": ['features.0', 'features.2', 'features.5', 'features.7', 'features.10', 'features.12', 'features.14', 'features.17', 'features.19', 'features.21', 'features.24', 'features.26', 'features.28'],
               "VGG16_bn": ['features.0', 'features.3', 'features.7', 'features.10', 'features.14', 'features.17', 'features.20', 'features.24', 'features.27', 'features.30', 'features.34', 'features.37', 'features.40'],
               "VGG19": ['features.0', 'features.2', 'features.5', 'features.7', 'features.10', 'features.12', 'features.14', 'features.16', 'features.19',  'features.21', 'features.23', 'features.25', 'features.28',  'features.30', 'features.34'],
               "VGG19_bn": ['features.0', 'features.3', 'features.7', 'features.10', 'features.14', 'features.17', 'features.20', 'features.23', 'features.27', 'features.30', 'features.33', 'features.36', 'features.40', 'features.43', 'features.46',  'features.49'],
               "Densenet121": ['features.denseblock1', 'features.denseblock2', 'features.denseblock3', 'features.denseblock4'],
               "Densenet161": ['features.denseblock1', 'features.denseblock2', 'features.denseblock3', 'features.denseblock4'],
               "Densenet169": ['features.denseblock1', 'features.denseblock2', 'features.denseblock3', 'features.denseblock4'],
               "Densenet201": ['features.denseblock1', 'features.denseblock2', 'features.denseblock3', 'features.denseblock4'],
               "GoogleNet": ['conv1', 'conv2', 'conv3', "inception3a", 'inception3b', 'inception4a', 'inception4b', 'inception4c', 'inception4d', 'inception4e', "inception5a", 'inception5b'],
               "ShuffleNetV2x05": ['conv1', 'stage2', 'stage3', 'stage4', 'conv5', 'fc'],
               "ShuffleNetV2x10": ['conv1', 'stage2', 'stage3', 'stage4', 'conv5', 'fc'],
               "mobilenet_v2": ['features', 'features.1', 'features.2', 'features.3', 'features.4', 'features.5', 'features.6', 'features.7', 'features.8', 'features.9', 'features.10', 'features.11', 'features.12', 'features.13', 'features.14', 'features.15', 'features.16', 'features.17', 'features.18'],
               "mobilenet_v3_large": ['features', 'features.0', 'features.1', 'features.2', 'features.3', 'features.4', 'features.5', 'features.6', 'features.7', 'features.8', 'features.9', 'features.10', 'features.11', 'features.12', 'features.13', 'features.14', 'features.15', 'features.16'],
               "mobilenet_v3_small": ['features', 'features.0', 'features.1', 'features.2', 'features.3', 'features.4', 'features.5', 'features.6', 'features.7', 'features.8', 'features.9', 'features.10', 'features.11', 'features.12'],
               "resnext50_32x4d": ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc'],
               "resnext101_32x8d": ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc'],
               "wide_resnet50_2": ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc'],
               "wide_resnet101_2": ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc'],
               "mnasnet05": ['layers', 'layers.0', 'layers.1', 'layers.2', 'layers.3', 'layers.4', 'layers.5', 'layers.6', 'layers.7', 'layers.8', 'layers.9', 'layers.10', 'layers.11', 'layers.12', 'layers.13', 'layers.14', 'layers.15', 'layers.16'],
               "mnasnet10": ['layers', 'layers.0', 'layers.1', 'layers.2', 'layers.3', 'layers.4', 'layers.5', 'layers.6', 'layers.7', 'layers.8', 'layers.9', 'layers.10', 'layers.11', 'layers.12', 'layers.13', 'layers.14', 'layers.15', 'layers.16'],
               "efficientnet_b0": ['features', 'features.0', 'features.1', 'features.2', 'features.3', 'features.4', 'features.5', 'features.6', 'features.7', 'features.8'],
               "efficientnet_b1": ['features', 'features.0', 'features.1', 'features.2', 'features.3', 'features.4', 'features.5', 'features.6', 'features.7', 'features.8'],
               "efficientnet_b2": ['features', 'features.0', 'features.1', 'features.2', 'features.3', 'features.4', 'features.5', 'features.6', 'features.7', 'features.8'],
               "efficientnet_b3": ['features', 'features.0', 'features.1', 'features.2', 'features.3', 'features.4', 'features.5', 'features.6', 'features.7', 'features.8'],
               "efficientnet_b4": ['features', 'features.0', 'features.1', 'features.2', 'features.3', 'features.4', 'features.5', 'features.6', 'features.7', 'features.8'],
               "efficientnet_b5": ['features', 'features.0', 'features.1', 'features.2', 'features.3', 'features.4', 'features.5', 'features.6', 'features.7', 'features.8'],
               "efficientnet_b6": ['features', 'features.0', 'features.1', 'features.2', 'features.3', 'features.4', 'features.5', 'features.6', 'features.7', 'features.8'],
               "efficientnet_b7": ['features', 'features.0', 'features.1', 'features.2', 'features.3', 'features.4', 'features.5', 'features.6', 'features.7', 'features.8'],
               "regnet_y_400mf": ['trunk_output', 'trunk_output.block1', 'trunk_output.block2', 'trunk_output.block3', 'trunk_output.block4', 'trunk_output.block2', 'fc'],
               "regnet_y_800mf": ['trunk_output', 'trunk_output.block1', 'trunk_output.block2', 'trunk_output.block3', 'trunk_output.block4', 'trunk_output.block2', 'fc'],
               "regnet_y_1_6gf": ['trunk_output', 'trunk_output.block1', 'trunk_output.block2', 'trunk_output.block3', 'trunk_output.block4', 'trunk_output.block2', 'fc'],
               "regnet_y_3_2gf": ['trunk_output', 'trunk_output.block1', 'trunk_output.block2', 'trunk_output.block3', 'trunk_output.block4', 'trunk_output.block2', 'fc'],
               "regnet_y_8gf": ['trunk_output', 'trunk_output.block1', 'trunk_output.block2', 'trunk_output.block3', 'trunk_output.block4', 'trunk_output.block2', 'fc'],
               "regnet_y_16gf": ['trunk_output', 'trunk_output.block1', 'trunk_output.block2', 'trunk_output.block3', 'trunk_output.block4', 'trunk_output.block2', 'fc'],
               "regnet_y_32gf": ['trunk_output', 'trunk_output.block1', 'trunk_output.block2', 'trunk_output.block3', 'trunk_output.block4', 'trunk_output.block2', 'fc'],
               "regnet_x_400mf": ['trunk_output', 'trunk_output.block1', 'trunk_output.block2', 'trunk_output.block3', 'trunk_output.block4', 'trunk_output.block2', 'fc'],
               "regnet_x_800mf": ['trunk_output', 'trunk_output.block1', 'trunk_output.block2', 'trunk_output.block3', 'trunk_output.block4', 'trunk_output.block2', 'fc'],
               "regnet_x_1_6gf": ['trunk_output', 'trunk_output.block1', 'trunk_output.block2', 'trunk_output.block3', 'trunk_output.block4', 'trunk_output.block2', 'fc'],
               "regnet_x_3_2gf": ['trunk_output', 'trunk_output.block1', 'trunk_output.block2', 'trunk_output.block3', 'trunk_output.block4', 'trunk_output.block2', 'fc'],
               "regnet_x_8gf": ['trunk_output', 'trunk_output.block1', 'trunk_output.block2', 'trunk_output.block3', 'trunk_output.block4', 'trunk_output.block2', 'fc'],
               "regnet_x_16gf": ['trunk_output', 'trunk_output.block1', 'trunk_output.block2', 'trunk_output.block3', 'trunk_output.block4', 'trunk_output.block2', 'fc'],
               "regnet_x_32gf": ['trunk_output', 'trunk_output.block1', 'trunk_output.block2', 'trunk_output.block3', 'trunk_output.block4', 'trunk_output.block2', 'fc']}


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
