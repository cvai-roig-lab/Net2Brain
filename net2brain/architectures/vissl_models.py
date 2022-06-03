import cv2
from torch.autograd import Variable as V
from torchvision import transforms as trn
import os.path as op
import os
from PIL import Image
import torch

import vissl
from vissl.models import build_model
from vissl.utils.hydra_config import compose_hydra_configuration, convert_to_attrdict
import urllib.request

MODELS = {'rn50_in1k_perm100_jigsaw': build_model,
          'rn50_in1k_perm10k_jigsaw': build_model,
          'rn50_in1k_perm2k_jigsaw': build_model,
          'rn50_in22k_perm2k_jigsaw': build_model,
          'rn50_in1k_jigsaw_goyal': build_model,
          'rn50_in22k_jigsaw_goyal': build_model,
          'rn50_yfcc100m_jigsaw_goyal': build_model,
          'rn50_in1k_npid_200ep_4kneg': build_model,
          'rn50_in1k_npid_oss': build_model,
          'rn50_in1k_rotnet': build_model,
          'rn50_in22k_rotnet': build_model,
          'rn50_in1k_clusterfit': build_model,
          'rn50_in1k_deepclusterv2_400ep_2x224': build_model,
          'rn50_in1k_deepclusterv2_400ep_2x160_4x96': build_model,
          'rn50_in1k_deepclusterv2_800ep_2x224_6x96': build_model,
          'rn50_in1k_simclr_100ep': build_model,
          'rn50_in1k_simclr_200ep': build_model,
          'rn50_in1k_simclr_400ep': build_model,
          'rn50_in1k_simclr_800ep': build_model,
          'rn50_in1k_simclr_1000ep': build_model,
          'rn50_in1k_swav_100ep_batch4k': build_model,
          'rn50_in1k_swav_200ep_batch4k': build_model,
          'rn50_in1k_swav_400ep_batch4k': build_model,
          'rn50_in1k_swav_800ep_batch4k': build_model,
          'rn50_in1k_swav_200ep_batch256': build_model,
          'rn50_in1k_swav_400ep_batch256': build_model,
          'rn50_in1k_swav_2x224_400ep_batch4k': build_model,
          'MoCoV2': build_model}


MODEL_NODES = {'rn50_in1k_perm100_jigsaw': ['trunk.base_model._feature_blocks.conv1', 'trunk.base_model._feature_blocks.layer1',
                                           'trunk.base_model._feature_blocks.layer2', 'trunk.base_model._feature_blocks.layer3', 
                                           'trunk.base_model._feature_blocks.layer4'],
                'rn50_in1k_perm10k_jigsaw': ['trunk.base_model._feature_blocks.conv1', 'trunk.base_model._feature_blocks.layer1',
                                           'trunk.base_model._feature_blocks.layer2', 'trunk.base_model._feature_blocks.layer3', 
                                           'trunk.base_model._feature_blocks.layer4'],
                'rn50_in1k_perm2k_jigsaw': ['trunk.base_model._feature_blocks.conv1', 'trunk.base_model._feature_blocks.layer1',
                                           'trunk.base_model._feature_blocks.layer2', 'trunk.base_model._feature_blocks.layer3', 
                                           'trunk.base_model._feature_blocks.layer4'],
                'rn50_in22k_perm2k_jigsaw': ['trunk.base_model._feature_blocks.conv1', 'trunk.base_model._feature_blocks.layer1',
                                           'trunk.base_model._feature_blocks.layer2', 'trunk.base_model._feature_blocks.layer3', 
                                           'trunk.base_model._feature_blocks.layer4'],
                'rn50_in1k_jigsaw_goyal': ['trunk.base_model._feature_blocks.conv1', 'trunk.base_model._feature_blocks.layer1',
                                           'trunk.base_model._feature_blocks.layer2', 'trunk.base_model._feature_blocks.layer3', 
                                           'trunk.base_model._feature_blocks.layer4'],
                'rn50_in22k_jigsaw_goyal': ['trunk.base_model._feature_blocks.conv1', 'trunk.base_model._feature_blocks.layer1',
                                           'trunk.base_model._feature_blocks.layer2', 'trunk.base_model._feature_blocks.layer3', 
                                           'trunk.base_model._feature_blocks.layer4'],
                'rn50_yfcc100m_jigsaw_goyal': ['trunk.base_model._feature_blocks.conv1', 'trunk.base_model._feature_blocks.layer1',
                                               'trunk.base_model._feature_blocks.layer2', 'trunk.base_model._feature_blocks.layer3',
                                               'trunk.base_model._feature_blocks.layer4'],
                'rn50_in1k_npid_200ep_4kneg': ['trunk.base_model._feature_blocks.conv1', 'trunk.base_model._feature_blocks.layer1',
                                               'trunk.base_model._feature_blocks.layer2', 'trunk.base_model._feature_blocks.layer3',
                                               'trunk.base_model._feature_blocks.layer4'],
                'rn50_in1k_npid_oss': ['trunk.base_model._feature_blocks.conv1', 'trunk.base_model._feature_blocks.layer1',
                                           'trunk.base_model._feature_blocks.layer2', 'trunk.base_model._feature_blocks.layer3',
                                           'trunk.base_model._feature_blocks.layer4'],
                'rn50_in1k_rotnet': ['trunk.base_model._feature_blocks.conv1', 'trunk.base_model._feature_blocks.layer1',
                                     'trunk.base_model._feature_blocks.layer2', 'trunk.base_model._feature_blocks.layer3',
                                     'trunk.base_model._feature_blocks.layer4'],
                'rn50_in22k_rotnet': ['trunk.base_model._feature_blocks.conv1', 'trunk.base_model._feature_blocks.layer1',
                                      'trunk.base_model._feature_blocks.layer2', 'trunk.base_model._feature_blocks.layer3',
                                      'trunk.base_model._feature_blocks.layer4'],
                'rn50_in1k_clusterfit': ['trunk.base_model._feature_blocks.conv1', 'trunk.base_model._feature_blocks.layer1',
                                         'trunk.base_model._feature_blocks.layer2', 'trunk.base_model._feature_blocks.layer3',
                                         'trunk.base_model._feature_blocks.layer4'],
                'rn50_in1k_deepclusterv2_400ep_2x224': ['trunk.base_model._feature_blocks.conv1', 'trunk.base_model._feature_blocks.layer1',
                                                        'trunk.base_model._feature_blocks.layer2', 'trunk.base_model._feature_blocks.layer3',
                                                        'trunk.base_model._feature_blocks.layer4'],
                'rn50_in1k_deepclusterv2_400ep_2x160_4x96': ['trunk.base_model._feature_blocks.conv1', 'trunk.base_model._feature_blocks.layer1',
                                                             'trunk.base_model._feature_blocks.layer2', 'trunk.base_model._feature_blocks.layer3',
                                                             'trunk.base_model._feature_blocks.layer4'],
                'rn50_in1k_deepclusterv2_800ep_2x224_6x96': ['trunk.base_model._feature_blocks.conv1', 'trunk.base_model._feature_blocks.layer1',
                                                             'trunk.base_model._feature_blocks.layer2', 'trunk.base_model._feature_blocks.layer3',
                                                             'trunk.base_model._feature_blocks.layer4'],
                'rn50_in1k_simclr_100ep': ['trunk.base_model._feature_blocks.conv1', 'trunk.base_model._feature_blocks.layer1',
                                           'trunk.base_model._feature_blocks.layer2', 'trunk.base_model._feature_blocks.layer3', 
                                           'trunk.base_model._feature_blocks.layer4'],
                'rn50_in1k_simclr_200ep': ['trunk.base_model._feature_blocks.conv1', 'trunk.base_model._feature_blocks.layer1',
                                           'trunk.base_model._feature_blocks.layer2', 'trunk.base_model._feature_blocks.layer3', 
                                           'trunk.base_model._feature_blocks.layer4'],
                'rn50_in1k_simclr_400ep': ['trunk.base_model._feature_blocks.conv1', 'trunk.base_model._feature_blocks.layer1',
                                           'trunk.base_model._feature_blocks.layer2', 'trunk.base_model._feature_blocks.layer3', 
                                           'trunk.base_model._feature_blocks.layer4'],
                'rn50_in1k_simclr_800ep': ['trunk.base_model._feature_blocks.conv1', 'trunk.base_model._feature_blocks.layer1',
                                           'trunk.base_model._feature_blocks.layer2', 'trunk.base_model._feature_blocks.layer3', 
                                           'trunk.base_model._feature_blocks.layer4'],
                'rn50_in1k_simclr_1000ep': ['trunk.base_model._feature_blocks.conv1', 'trunk.base_model._feature_blocks.layer1',
                                           'trunk.base_model._feature_blocks.layer2', 'trunk.base_model._feature_blocks.layer3', 
                                           'trunk.base_model._feature_blocks.layer4'],
                'rn50_in1k_swav_100ep_batch4k': ['trunk.base_model._feature_blocks.conv1', 'trunk.base_model._feature_blocks.layer1',
                                                 'trunk.base_model._feature_blocks.layer2', 'trunk.base_model._feature_blocks.layer3',
                                                 'trunk.base_model._feature_blocks.layer4'],
                'rn50_in1k_swav_200ep_batch4k': ['trunk.base_model._feature_blocks.conv1', 'trunk.base_model._feature_blocks.layer1',
                                                 'trunk.base_model._feature_blocks.layer2', 'trunk.base_model._feature_blocks.layer3',
                                                 'trunk.base_model._feature_blocks.layer4'],
                'rn50_in1k_swav_400ep_batch4k': ['trunk.base_model._feature_blocks.conv1', 'trunk.base_model._feature_blocks.layer1',
                                                 'trunk.base_model._feature_blocks.layer2', 'trunk.base_model._feature_blocks.layer3',
                                                 'trunk.base_model._feature_blocks.layer4'],
                'rn50_in1k_swav_800ep_batch4k': ['trunk.base_model._feature_blocks.conv1', 'trunk.base_model._feature_blocks.layer1',
                                                 'trunk.base_model._feature_blocks.layer2', 'trunk.base_model._feature_blocks.layer3',
                                                 'trunk.base_model._feature_blocks.layer4'],
                'rn50_in1k_swav_200ep_batch256': ['trunk.base_model._feature_blocks.conv1', 'trunk.base_model._feature_blocks.layer1',
                                                  'trunk.base_model._feature_blocks.layer2', 'trunk.base_model._feature_blocks.layer3',
                                                  'trunk.base_model._feature_blocks.layer4'],
                'rn50_in1k_swav_400ep_batch256': ['trunk.base_model._feature_blocks.conv1', 'trunk.base_model._feature_blocks.layer1',
                                                  'trunk.base_model._feature_blocks.layer2', 'trunk.base_model._feature_blocks.layer3',
                                                  'trunk.base_model._feature_blocks.layer4'],
                'rn50_in1k_swav_2x224_400ep_batch4k': ['trunk.base_model._feature_blocks.conv1', 'trunk.base_model._feature_blocks.layer1',
                                                       'trunk.base_model._feature_blocks.layer2', 'trunk.base_model._feature_blocks.layer3',
                                                       'trunk.base_model._feature_blocks.layer4'],
                'MoCoV2': ['trunk.base_model._feature_blocks.conv1', 'trunk.base_model._feature_blocks.layer1',
                           'trunk.base_model._feature_blocks.layer2', 'trunk.base_model._feature_blocks.layer3',
                           'trunk.base_model._feature_blocks.layer4']}


cfg_yamls = {'rn50_in1k_perm100_jigsaw': 'pretrain/jigsaw/jigsaw_8gpu_resnet.yaml',
             'rn50_in1k_perm10k_jigsaw': 'pretrain/jigsaw/jigsaw_8gpu_resnet.yaml',
             'rn50_in1k_perm2k_jigsaw': 'pretrain/jigsaw/jigsaw_8gpu_resnet.yaml',
             'rn50_in22k_perm2k_jigsaw': 'pretrain/jigsaw/jigsaw_8gpu_resnet.yaml',
             'rn50_in1k_jigsaw_goyal': 'pretrain/jigsaw/jigsaw_8gpu_resnet.yaml',
             'rn50_in22k_jigsaw_goyal': 'pretrain/jigsaw/jigsaw_8gpu_resnet.yaml',
             'rn50_yfcc100m_jigsaw_goyal': 'pretrain/jigsaw/jigsaw_8gpu_resnet.yaml',
             'rn50_in1k_npid_200ep_4kneg': 'pretrain/npid/npid_8gpu_resnet.yaml',
             'rn50_in1k_npid_oss': 'pretrain/npid/npid_8gpu_resnet.yaml',
             'rn50_in1k_rotnet': 'pretrain/rotnet/rotnet_8gpu_resnet.yaml',
             'rn50_in22k_rotnet': 'pretrain/rotnet/rotnet_8gpu_resnet.yaml',
             'rn50_in1k_clusterfit': 'pretrain/npid/npid_8gpu_resnet.yaml',
             'rn50_in1k_deepclusterv2_400ep_2x224': 'pretrain/deepcluster_v2/deepclusterv2_2crops_resnet.yaml',
             'rn50_in1k_deepclusterv2_400ep_2x160_4x96': 'pretrain/deepcluster_v2/deepclusterv2_2crops_resnet.yaml',
             'rn50_in1k_deepclusterv2_800ep_2x224_6x96': 'pretrain/deepcluster_v2/deepclusterv2_2crops_resnet.yaml',
             'rn50_in1k_simclr_100ep': 'pretrain/simclr/simclr_8node_resnet.yaml',
             'rn50_in1k_simclr_200ep': 'pretrain/simclr/simclr_8node_resnet.yaml',
             'rn50_in1k_simclr_400ep': 'pretrain/simclr/simclr_8node_resnet.yaml',
             'rn50_in1k_simclr_800ep': 'pretrain/simclr/simclr_8node_resnet.yaml',
             'rn50_in1k_simclr_1000ep': 'pretrain/simclr/simclr_8node_resnet.yaml',
             'rn50_in1k_swav_100ep_batch4k': 'pretrain/swav/swav_8node_resnet.yaml',
             'rn50_in1k_swav_200ep_batch4k': 'pretrain/swav/swav_8node_resnet.yaml',
             'rn50_in1k_swav_400ep_batch4k': 'pretrain/swav/swav_8node_resnet.yaml',
             'rn50_in1k_swav_800ep_batch4k': 'pretrain/swav/swav_8node_resnet.yaml',
             'rn50_in1k_swav_200ep_batch256': 'pretrain/swav/swav_8node_resnet.yaml',
             'rn50_in1k_swav_400ep_batch256': 'pretrain/swav/swav_8node_resnet.yaml',
             'rn50_in1k_swav_2x224_400ep_batch4k': 'pretrain/swav/swav_8node_resnet.yaml',
             'MoCoV2': 'pretrain/moco/moco_1node_resnet.yaml'}


cfg_checkpoints = {'rn50_in1k_perm100_jigsaw': "https://dl.fbaipublicfiles.com/vissl/model_zoo/jigsaw_rn50_in1k_ep105_perm2k_jigsaw_8gpu_resnet_17_07_20.db174a43/model_final_checkpoint_phase104.torch",
                   'rn50_in1k_perm10k_jigsaw': "https://dl.fbaipublicfiles.com/vissl/model_zoo/jigsaw_rn50_in1k_ep105_perm2k_jigsaw_8gpu_resnet_20_07_20.3d706467/model_final_checkpoint_phase104.torch",
                   'rn50_in1k_perm2k_jigsaw': "https://dl.fbaipublicfiles.com/vissl/model_zoo/jigsaw_rn50_in1k_ep105_perm2k_jigsaw_8gpu_resnet_17_07_20.cccee144/model_final_checkpoint_phase104.torch",
                   'rn50_in22k_perm2k_jigsaw': "https://dl.fbaipublicfiles.com/vissl/model_zoo/converted_vissl_jigsaw_rn50_perm2k_in22k_8gpu_ep105.torch",
                   'rn50_in1k_jigsaw_goyal': "https://dl.fbaipublicfiles.com/vissl/model_zoo/converted_vissl_rn50_jigsaw_in1k_goyal19.torch",
                   'rn50_in22k_jigsaw_goyal': "https://dl.fbaipublicfiles.com/vissl/model_zoo/converted_vissl_rn50_jigsaw_in22k_goyal19.torch",
                   'rn50_yfcc100m_jigsaw_goyal': "https://dl.fbaipublicfiles.com/vissl/model_zoo/converted_vissl_rn50_jigsaw_yfcc100m_goyal19.torch",
                   'rn50_in1k_npid_200ep_4kneg': "https://dl.fbaipublicfiles.com/vissl/model_zoo/npid_1node_200ep_4kneg_npid_8gpu_resnet_23_07_20.9eb36512/model_final_checkpoint_phase199.torch",
                   'rn50_in1k_npid_oss': "https://dl.fbaipublicfiles.com/vissl/model_zoo/converted_vissl_rn50_npid_lemniscate_neg4k_stepLR_8gpu.torch",
                   'rn50_in1k_rotnet': "https://dl.fbaipublicfiles.com/vissl/model_zoo/rotnet_rn50_in1k_ep105_rotnet_8gpu_resnet_17_07_20.46bada9f/model_final_checkpoint_phase125.torch",
                   'rn50_in22k_rotnet': "https://dl.fbaipublicfiles.com/vissl/model_zoo/converted_vissl_rn50_rotnet_in22k_ep105.torch",
                   'rn50_in1k_clusterfit': "https://dl.fbaipublicfiles.com/vissl/model_zoo/converted_vissl_rn50_rotnet_16kclusters_in1k_ep105.torch",
                   'rn50_in1k_deepclusterv2_400ep_2x224': "https://dl.fbaipublicfiles.com/vissl/model_zoo/deepclusterv2_400ep_2x224_pretrain.pth.tar",
                   'rn50_in1k_deepclusterv2_400ep_2x160_4x96': "https://dl.fbaipublicfiles.com/vissl/model_zoo/deepclusterv2_400ep_pretrain.pth.tar",
                   'rn50_in1k_deepclusterv2_800ep_2x224_6x96': "https://dl.fbaipublicfiles.com/vissl/model_zoo/deepclusterv2_800ep_pretrain.pth.tar",
                   'rn50_in1k_simclr_100ep': "https://dl.fbaipublicfiles.com/vissl/model_zoo/simclr_rn50_100ep_simclr_8node_resnet_16_07_20.8edb093e/model_final_checkpoint_phase99.torch",
                   'rn50_in1k_simclr_200ep': "https://dl.fbaipublicfiles.com/vissl/model_zoo/simclr_rn50_200ep_simclr_8node_resnet_16_07_20.a816c0ef/model_final_checkpoint_phase199.torch",
                   'rn50_in1k_simclr_400ep': "https://dl.fbaipublicfiles.com/vissl/model_zoo/simclr_rn50_400ep_simclr_8node_resnet_16_07_20.36b338ef/model_final_checkpoint_phase399.torch",
                   'rn50_in1k_simclr_800ep': "https://dl.fbaipublicfiles.com/vissl/model_zoo/simclr_rn50_800ep_simclr_8node_resnet_16_07_20.7e8feed1/model_final_checkpoint_phase799.torch",
                   'rn50_in1k_simclr_1000ep': "https://dl.fbaipublicfiles.com/vissl/model_zoo/simclr_rn50_1000ep_simclr_8node_resnet_16_07_20.afe428c7/model_final_checkpoint_phase999.torch",
                   'rn50_in1k_swav_100ep_batch4k': "https://dl.fbaipublicfiles.com/vissl/model_zoo/swav_in1k_rn50_100ep_swav_8node_resnet_27_07_20.7e6fc6bf/model_final_checkpoint_phase99.torch",
                   'rn50_in1k_swav_200ep_batch4k': "https://dl.fbaipublicfiles.com/vissl/model_zoo/swav_in1k_rn50_200ep_swav_8node_resnet_27_07_20.bd595bb0/model_final_checkpoint_phase199.torch",
                   'rn50_in1k_swav_400ep_batch4k': "https://dl.fbaipublicfiles.com/vissl/model_zoo/swav_in1k_rn50_400ep_swav_8node_resnet_27_07_20.a5990fc9/model_final_checkpoint_phase399.torch",
                   'rn50_in1k_swav_800ep_batch4k': "https://dl.fbaipublicfiles.com/vissl/model_zoo/swav_in1k_rn50_800ep_swav_8node_resnet_27_07_20.a0a6b676/model_final_checkpoint_phase799.torch",
                   'rn50_in1k_swav_200ep_batch256': "https://dl.fbaipublicfiles.com/vissl/model_zoo/swav_4gpu_bs64_200ep_2x224_6x96_queue_swav_8node_resnet_28_07_20.a8f2c735/model_final_checkpoint_phase199.torch",
                   'rn50_in1k_swav_400ep_batch256': "https://dl.fbaipublicfiles.com/vissl/model_zoo/swav_4gpu_bs64_400ep_2x224_6x96_queue_swav_8node_resnet_28_07_20.5e967ca0/model_final_checkpoint_phase399.torch",
                   'rn50_in1k_swav_2x224_400ep_batch4k': "https://dl.fbaipublicfiles.com/vissl/model_zoo/swav_8node_2x224_rn50_in1k_swav_8node_resnet_30_07_20.c8fd7169/model_final_checkpoint_phase399.torch",
                   'MoCoV2': "https://dl.fbaipublicfiles.com/vissl/model_zoo/moco_v2_1node_lr.03_step_b32_zero_init/model_final_checkpoint_phase199.torch"}



def download_weights(model_name):
    CURRENT_DIR = op.abspath(os.curdir)
    file_location = os.path.join(CURRENT_DIR, "configs")
    
    if not os.path.exists(file_location):
        os.makedirs(file_location)
        
    checkpoint_link = cfg_checkpoints[model_name]
    file_name = checkpoint_link.split("/")[-1]
    file_path = os.path.join(file_location, file_name)
    
    if not os.path.exists(file_path):
        urllib.request.urlretrieve(checkpoint_link, file_path)
    
    return file_path
        
    

def configurator(model_name):
    print("in config")
    
    file_path = download_weights(model_name)
    print("make config")
    cfg = [
        'config=' + cfg_yamls[model_name],
        'config.MODEL.WEIGHTS_INIT.PARAMS_FILE=' + file_path,
        'config.MODEL.FEATURE_EVAL_SETTINGS.EVAL_MODE_ON=True',
        'config.MODEL.FEATURE_EVAL_SETTINGS.FREEZE_TRUNK_ONLY=True',
        'config.MODEL.FEATURE_EVAL_SETTINGS.EXTRACT_TRUNK_FEATURES_ONLY=True',
        'config.MODEL.FEATURE_EVAL_SETTINGS.SHOULD_FLATTEN_FEATS=False',
        'config.MODEL.FEATURE_EVAL_SETTINGS.LINEAR_EVAL_FEAT_POOL_OPS_MAP=[["res5avg", ["Identity", []]]]'
    ]

    cfg = compose_hydra_configuration(cfg)
    _, cfg = convert_to_attrdict(cfg)
    
    return cfg



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
