import cv2
from torch.autograd import Variable as V
from torchvision import transforms as trn
from PIL import Image
import visualpriors
import torch

# Colorization is missing, because the model architecture needs to change to load the state dict...
#         'colorization': visualpriors.taskonomy_network.TaskonomyEncoder,
MODELS = {'autoencoding': visualpriors.taskonomy_network.TaskonomyEncoder,
          'curvature': visualpriors.taskonomy_network.TaskonomyEncoder,
          'class_object': visualpriors.taskonomy_network.TaskonomyEncoder,
          'class_scene': visualpriors.taskonomy_network.TaskonomyEncoder,
          'denoising': visualpriors.taskonomy_network.TaskonomyEncoder,
          'depth_euclidean': visualpriors.taskonomy_network.TaskonomyEncoder,
          'edge_occlusion': visualpriors.taskonomy_network.TaskonomyEncoder,
          'edge_texture': visualpriors.taskonomy_network.TaskonomyEncoder,
          'egomotion': visualpriors.taskonomy_network.TaskonomyEncoder,
          'fixated_pose': visualpriors.taskonomy_network.TaskonomyEncoder,
          'inpainting': visualpriors.taskonomy_network.TaskonomyEncoder,
          'jigsaw': visualpriors.taskonomy_network.TaskonomyEncoder,
          'keypoints2d': visualpriors.taskonomy_network.TaskonomyEncoder,
          'keypoints3d': visualpriors.taskonomy_network.TaskonomyEncoder,
          'nonfixated_pose': visualpriors.taskonomy_network.TaskonomyEncoder,
          'normal': visualpriors.taskonomy_network.TaskonomyEncoder,
          'point_matching': visualpriors.taskonomy_network.TaskonomyEncoder,
          'reshading': visualpriors.taskonomy_network.TaskonomyEncoder,
          'room_layout': visualpriors.taskonomy_network.TaskonomyEncoder,
          'segment_unsup2d': visualpriors.taskonomy_network.TaskonomyEncoder,
          'segment_unsup25d': visualpriors.taskonomy_network.TaskonomyEncoder,
          'segment_semantic': visualpriors.taskonomy_network.TaskonomyEncoder,
          'vanishing_point': visualpriors.taskonomy_network.TaskonomyEncoder}

MODEL_NODES = {'autoencoding': ['layer1', 'layer2', 'layer3', 'layer4'],
               'curvature': ['layer1', 'layer2', 'layer3', 'layer4'],
               'class_object': ['layer1', 'layer2', 'layer3', 'layer4'],
               'class_scene': ['layer1', 'layer2', 'layer3', 'layer4'],
               'denoising': ['layer1', 'layer2', 'layer3', 'layer4'],
               'depth_euclidean': ['layer1', 'layer2', 'layer3', 'layer4'],
               'edge_occlusion': ['layer1', 'layer2', 'layer3', 'layer4'],
               'edge_texture': ['layer1', 'layer2', 'layer3', 'layer4'],
               'egomotion': ['layer1', 'layer2', 'layer3', 'layer4'],
               'fixated_pose': ['layer1', 'layer2', 'layer3', 'layer4'],
               'inpainting': ['layer1', 'layer2', 'layer3', 'layer4'],
               'jigsaw': ['layer1', 'layer2', 'layer3', 'layer4'],
               'keypoints2d': ['layer1', 'layer2', 'layer3', 'layer4'],
               'keypoints3d': ['layer1', 'layer2', 'layer3', 'layer4'],
               'nonfixated_pose': ['layer1', 'layer2', 'layer3', 'layer4'],
               'normal': ['layer1', 'layer2', 'layer3', 'layer4'],
               'point_matching': ['layer1', 'layer2', 'layer3', 'layer4'],
               'reshading': ['layer1', 'layer2', 'layer3', 'layer4'],
               'room_layout': ['layer1', 'layer2', 'layer3', 'layer4'],
               'segment_unsup2d': ['layer1', 'layer2', 'layer3', 'layer4'],
               'segment_unsup25d': ['layer1', 'layer2', 'layer3', 'layer4'],
               'segment_semantic': ['layer1', 'layer2', 'layer3', 'layer4'],
               'vanishing_point': ['layer1', 'layer2', 'layer3', 'layer4']}

MODEL_WEIGHTS = {'autoencoding': 'https://github.com/alexsax/visual-prior/raw/networks/assets/pytorch/autoencoding_encoder-e35146c09253720e97c0a7f8ee4e896ac931f5faa1449df003d81e6089ac6307.pth',
                 'colorization': 'https://github.com/alexsax/visual-prior/raw/networks/assets/pytorch/colorization_encoder-5ed817acdd28d13e443d98ad15ebe1c3059a3252396a2dff6a2090f6f86616a5.pth',
                 'curvature': 'https://github.com/alexsax/visual-prior/raw/networks/assets/pytorch/curvature_encoder-3767cf5d06d9c6bca859631eb5a3c368d66abeb15542171b94188ffbe47d7571.pth',
                 'class_object': 'https://github.com/alexsax/visual-prior/raw/networks/assets/pytorch/class_object_encoder-4a4e42dad58066039a0d2f9d128bb32e93a7e4aa52edb2d2a07bcdd1a6536c18.pth',
                 'class_scene': 'https://github.com/alexsax/visual-prior/raw/networks/assets/pytorch/class_scene_encoder-ad85764467cddafd98211313ceddebb98adf2a6bee2cedfe0b922a37ae65eaf8.pth',
                 'denoising': 'https://github.com/alexsax/visual-prior/raw/networks/assets/pytorch/denoising_encoder-b64cab95af4a2c565066a7e8effaf37d6586c3b9389b47fff9376478d849db38.pth',
                 'depth_euclidean': 'https://github.com/alexsax/visual-prior/raw/networks/assets/pytorch/depth_euclidean_encoder-88f18d41313de7dbc88314a7f0feec3023047303d94d73eb8622dc40334ef149.pth',
                 'edge_occlusion': 'https://github.com/alexsax/visual-prior/raw/networks/assets/pytorch/edge_occlusion_encoder-5ac3f3e918131f61e01fe95e49f462ae2fc56aa463f8d353ca84cd4e248b9c08.pth',
                 'edge_texture': 'https://github.com/alexsax/visual-prior/raw/networks/assets/pytorch/edge_texture_encoder-be2d686a6a4dfebe968d16146a17176eba37e29f736d5cd9a714317c93718810.pth',
                 'egomotion': 'https://github.com/alexsax/visual-prior/raw/networks/assets/pytorch/egomotion_encoder-9aa647c34bf98f9e491e0b37890d77566f6ae35ccf41d9375c674511318d571c.pth',
                 'fixated_pose': 'https://github.com/alexsax/visual-prior/raw/networks/assets/pytorch/fixated_pose_encoder-78cf321518abc16f9f4782b9e5d4e8f5d6966c373d951928a26f872e55297567.pth',
                 'inpainting': 'https://github.com/alexsax/visual-prior/raw/networks/assets/pytorch/inpainting_encoder-bf96fbaaea9268a820a19a1d13dbf6af31798f8983c6d9203c00fab2d236a142.pth',
                 'jigsaw': 'https://github.com/alexsax/visual-prior/raw/networks/assets/pytorch/jigsaw_encoder-0c2b342c9080f8713c178b04aa6c581ed3a0112fecaf78edc4c04e0a90516e39.pth',
                 'keypoints2d': 'https://github.com/alexsax/visual-prior/raw/networks/assets/pytorch/keypoints2d_encoder-6b77695acff4c84091c484a7b128a1e28a7e9c36243eda278598f582cf667fe0.pth',
                 'keypoints3d': 'https://github.com/alexsax/visual-prior/raw/networks/assets/pytorch/keypoints3d_encoder-7e3f1ec97b82ae30030b7ea4fec2dc606b71497d8c0335d05f0be3dc909d000d.pth',
                 'nonfixated_pose': 'https://github.com/alexsax/visual-prior/raw/networks/assets/pytorch/nonfixated_pose_encoder-3433a600ca9ff384b9898e55d86a186d572c2ebbe4701489a373933e3cfd5b8b.pth',
                 'normal': 'https://github.com/alexsax/visual-prior/raw/networks/assets/pytorch/normal_encoder-f5e2c7737e4948e3b2a822f584892c342eaabbe66661576ba50db7cdd40561c5.pth',
                 'point_matching': 'https://github.com/alexsax/visual-prior/raw/networks/assets/pytorch/point_matching_encoder-4bd2a6b2909d9998fabaf0278ab568f42f2b692a648e28555ede6c6cda5361f4.pth',
                 'reshading': 'https://github.com/alexsax/visual-prior/raw/networks/assets/pytorch/reshading_encoder-de456246e171dc8407fb2951539aa60d75925ae0f1dbb43f110b7768398b36a6.pth',
                 'room_layout': 'https://github.com/alexsax/visual-prior/raw/networks/assets/pytorch/room_layout_encoder-1e1662f43b834261464b1825227a04efba59b50cc8883bee9adc3ddafd4796c1.pth',
                 'segment_unsup2d': 'https://github.com/alexsax/visual-prior/raw/networks/assets/pytorch/segment_unsup2d_encoder-b679053a920e8bcabf0cd454606098ae85341e054080f2be29473971d4265964.pth',
                 'segment_unsup25d': 'https://github.com/alexsax/visual-prior/raw/networks/assets/pytorch/segment_unsup25d_encoder-7d12d2500c18c003ffc23943214f5dfd74932f0e3d03dde2c3a81ebc406e31a0.pth',
                 'segment_semantic': 'https://github.com/alexsax/visual-prior/raw/networks/assets/pytorch/segment_semantic_encoder-bb3007244520fc89cd111e099744a22b1e5c98cd83ed3f116fbff6376d220127.pth',
                 'vanishing_point': 'https://github.com/alexsax/visual-prior/raw/networks/assets/pytorch/vanishing_point_encoder-afd2ae9b71d46a54efc5231b3e38ebc3e35bfab78cb0a78d9b75863a240b19a8.pth'}


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
    
    centre_crop_grey = trn.Compose([  # For grayscale images
        trn.Resize((224, 224)),
        trn.ToTensor()
    ])

    image = Image.open(image)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model_name == 'colorization':
        image = image.convert('L')
        image = V(centre_crop_grey(image).unsqueeze(0))
        
        # To Cuda
        if device == torch.device('cuda'):  # send to cuda
            image = image.cuda()
        
        return image
    else:
        image = V(centre_crop(image).unsqueeze(0))
        
        # To Cuda
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
    
    centre_crop_grey = trn.Compose([  # For grayscale images
        trn.Resize((224, 224)),
        trn.ToTensor()
    ])

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    if model_name == 'colorization':
        pil_image = pil_image.convert('L')
        pil_image = V(centre_crop_grey(pil_image).unsqueeze(0))
        
        if device == torch.device('cuda'):  # send to cuda
            pil_image = pil_image.cuda()
            
        return pil_image
    else:
        pil_image = V(centre_crop(pil_image).unsqueeze(0))
        
        if device == torch.device('cuda'):  # send to cuda
            pil_image = pil_image.cuda()
            
        return pil_image
