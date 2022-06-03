#https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=wuj3vR90Gmur


import cv2
from torch.autograd import Variable as V
from torchvision import transforms as trn
from PIL import Image
import torch
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.modeling import build_model

MODELS = {'COCO-Detection_-_faster_rcnn_R_50_C4_1x.yaml': build_model,
          'COCO-Detection_-_faster_rcnn_R_50_DC5_1x.yaml': build_model,
          'COCO-Detection_-_faster_rcnn_R_50_FPN_1x.yaml': build_model,
          'COCO-Detection_-_faster_rcnn_R_50_C4_3x.yaml': build_model,
          'COCO-Detection_-_faster_rcnn_R_50_DC5_3x.yaml': build_model,
          'COCO-Detection_-_faster_rcnn_R_50_FPN_3x.yaml': build_model,
          'COCO-Detection_-_faster_rcnn_R_101_C4_3x.yaml': build_model,
          'COCO-Detection_-_faster_rcnn_R_101_DC5_3x.yaml': build_model,
          'COCO-Detection_-_faster_rcnn_R_101_FPN_3x.yaml': build_model,
          'COCO-Detection_-_faster_rcnn_X_101_32x8d_FPN_3x.yaml': build_model,
          'COCO-Detection_-_retinanet_R_50_FPN_1x.yaml': build_model,
          'COCO-Detection_-_retinanet_R_50_FPN_3x.yaml': build_model,
          'COCO-Detection_-_retinanet_R_101_FPN_3x.yaml': build_model,
          'COCO-Detection_-_rpn_R_50_C4_1x.yaml': build_model,
          'COCO-Detection_-_rpn_R_50_FPN_1x.yaml': build_model,
          'COCO-InstanceSegmentation_-_mask_rcnn_R_50_C4_1x.yaml': build_model,
          'COCO-InstanceSegmentation_-_mask_rcnn_R_50_DC5_1x.yaml': build_model,
          'COCO-InstanceSegmentation_-_mask_rcnn_R_50_FPN_1x.yaml': build_model,
          'COCO-InstanceSegmentation_-_mask_rcnn_R_50_C4_3x.yaml': build_model,
          'COCO-InstanceSegmentation_-_mask_rcnn_R_50_DC5_3x.yaml': build_model,
          'COCO-InstanceSegmentation_-_mask_rcnn_R_50_FPN_3x.yaml': build_model,
          'COCO-InstanceSegmentation_-_mask_rcnn_R_101_C4_3x.yaml': build_model,
          'COCO-InstanceSegmentation_-_mask_rcnn_R_101_DC5_3x.yaml': build_model,
          'COCO-InstanceSegmentation_-_mask_rcnn_R_101_FPN_3x.yaml': build_model,
          'COCO-InstanceSegmentation_-_mask_rcnn_X_101_32x8d_FPN_3x.yaml': build_model,
          'COCO-Keypoints_-_keypoint_rcnn_R_50_FPN_1x.yaml': build_model,
          'COCO-Keypoints_-_keypoint_rcnn_R_101_FPN_3x.yaml': build_model,
          'COCO-Keypoints_-_keypoint_rcnn_X_101_32x8d_FPN_3x.yaml': build_model,
          'COCO-PanopticSegmentation_-_panoptic_fpn_R_50_1x.yaml': build_model,
          'COCO-PanopticSegmentation_-_panoptic_fpn_R_50_3x.yaml': build_model,
          'COCO-PanopticSegmentation_-_panoptic_fpn_R_101_3x.yaml': build_model,
          'LVISv0.5-InstanceSegmentation_-_mask_rcnn_R_50_FPN_1x.yaml': build_model,
          'LVISv0.5-InstanceSegmentation_-_mask_rcnn_R_101_FPN_1x.yaml': build_model,
          'LVISv0.5-InstanceSegmentation_-_mask_rcnn_X_101_32x8d_FPN_1x.yaml': build_model,
          'Cityscapes_-_mask_rcnn_R_50_FPN.yaml': build_model,
          'PascalVOC-Detection_-_faster_rcnn_R_50_C4.yaml': build_model,
          'Misc_-_mask_rcnn_R_50_FPN_1x_dconv_c3-c5.yaml': build_model,
          'Misc_-_mask_rcnn_R_50_FPN_3x_dconv_c3-c5.yaml': build_model,
          'Misc_-_cascade_mask_rcnn_R_50_FPN_1x.yaml': build_model,
          'Misc_-_cascade_mask_rcnn_R_50_FPN_3x.yaml': build_model,
          'Misc_-_mask_rcnn_R_50_FPN_3x_syncbn.yaml': build_model,
          'Misc_-_mask_rcnn_R_50_FPN_3x_gn.yaml': build_model,
          'Misc_-_scratch_mask_rcnn_R_50_FPN_3x_gn.yaml': build_model,
          'Misc_-_scratch_mask_rcnn_R_50_FPN_9x_gn.yaml': build_model,
          'Misc_-_scratch_mask_rcnn_R_50_FPN_9x_syncbn.yaml': build_model,
          'Misc_-_panoptic_fpn_R_101_dconv_cascade_gn_3x.yaml': build_model,
          'Misc_-_cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml': build_model,
          'Detectron1-Comparisons_-_faster_rcnn_R_50_FPN_noaug_1x.yaml': build_model,
          'Detectron1-Comparisons_-_mask_rcnn_R_50_FPN_noaug_1x.yaml': build_model,
          'Detectron1-Comparisons_-_keypoint_rcnn_R_50_FPN_1x.yaml': build_model}


MODEL_NODES = {'COCO-Detection_-_faster_rcnn_R_50_C4_1x.yaml': ['backbone'],
               'COCO-Detection_-_faster_rcnn_R_50_DC5_1x.yaml': ['backbone'],
               'COCO-Detection_-_faster_rcnn_R_50_FPN_1x.yaml': ['backbone'],
               'COCO-Detection_-_faster_rcnn_R_50_C4_3x.yaml': ['backbone'],
               'COCO-Detection_-_faster_rcnn_R_50_DC5_3x.yaml': ['backbone'],
               'COCO-Detection_-_faster_rcnn_R_50_FPN_3x.yaml': ['backbone', 'backbone.bottom_up'],
               'COCO-Detection_-_faster_rcnn_R_101_C4_3x.yaml': ['backbone'],
               'COCO-Detection_-_faster_rcnn_R_101_DC5_3x.yaml': ['backbone'],
               'COCO-Detection_-_faster_rcnn_R_101_FPN_3x.yaml': ['backbone', 'backbone.bottom_up'],
               'COCO-Detection_-_faster_rcnn_X_101_32x8d_FPN_3x.yaml': ['backbone', 'backbone.bottom_up'],
               'COCO-Detection_-_retinanet_R_50_FPN_1x.yaml': ['backbone', 'backbone.bottom_up'],
               'COCO-Detection_-_retinanet_R_50_FPN_3x.yaml': ['backbone', 'backbone.bottom_up'],
               'COCO-Detection_-_retinanet_R_101_FPN_3x.yaml': ['backbone', 'backbone.bottom_up'],
               'COCO-Detection_-_rpn_R_50_C4_1x.yaml': ['backbone'],
               'COCO-Detection_-_rpn_R_50_FPN_1x.yaml': ['backbone', 'backbone.bottom_up'],
               'COCO-InstanceSegmentation_-_mask_rcnn_R_50_C4_1x.yaml': ['backbone'],
               'COCO-InstanceSegmentation_-_mask_rcnn_R_50_DC5_1x.yaml': ['backbone'],
               'COCO-InstanceSegmentation_-_mask_rcnn_R_50_FPN_1x.yaml': ['backbone', 'backbone.bottom_up'],
               'COCO-InstanceSegmentation_-_mask_rcnn_R_50_C4_3x.yaml': ['backbone'],
               'COCO-InstanceSegmentation_-_mask_rcnn_R_50_DC5_3x.yaml': ['backbone'],
               'COCO-InstanceSegmentation_-_mask_rcnn_R_50_FPN_3x.yaml': ['backbone', 'backbone.bottom_up'],
               'COCO-InstanceSegmentation_-_mask_rcnn_R_101_C4_3x.yaml': ['backbone'],
               'COCO-InstanceSegmentation_-_mask_rcnn_R_101_DC5_3x.yaml': ['backbone'],
               'COCO-InstanceSegmentation_-_mask_rcnn_R_101_FPN_3x.yaml': ['backbone', 'backbone.bottom_up'],
               'COCO-InstanceSegmentation_-_mask_rcnn_X_101_32x8d_FPN_3x.yaml': ['backbone', 'backbone.bottom_up'],
               'COCO-Keypoints_-_keypoint_rcnn_R_50_FPN_1x.yaml': ['backbone', 'backbone.bottom_up'],
               'COCO-Keypoints_-_keypoint_rcnn_R_101_FPN_3x.yaml': ['backbone', 'backbone.bottom_up'],
               'COCO-Keypoints_-_keypoint_rcnn_X_101_32x8d_FPN_3x.yaml': ['backbone', 'backbone.bottom_up'],
               'COCO-PanopticSegmentation_-_panoptic_fpn_R_50_1x.yaml': ['backbone', 'backbone.bottom_up'],
               'COCO-PanopticSegmentation_-_panoptic_fpn_R_50_3x.yaml': ['backbone', 'backbone.bottom_up'],
               'COCO-PanopticSegmentation_-_panoptic_fpn_R_101_3x.yaml': ['backbone', 'backbone.bottom_up'],
               'LVISv0.5-InstanceSegmentation_-_mask_rcnn_R_50_FPN_1x.yaml': ['backbone', 'backbone.bottom_up'],
               'LVISv0.5-InstanceSegmentation_-_mask_rcnn_R_101_FPN_1x.yaml': ['backbone', 'backbone.bottom_up'],
               'LVISv0.5-InstanceSegmentation_-_mask_rcnn_X_101_32x8d_FPN_1x.yaml': ['backbone', 'backbone.bottom_up'],
               'Cityscapes_-_mask_rcnn_R_50_FPN.yaml': ['backbone', 'backbone.bottom_up'],
               'PascalVOC-Detection_-_faster_rcnn_R_50_C4.yaml': ['backbone'],
               'Misc_-_mask_rcnn_R_50_FPN_1x_dconv_c3-c5.yaml': ['backbone', 'backbone.bottom_up'],
               'Misc_-_mask_rcnn_R_50_FPN_3x_dconv_c3-c5.yaml': ['backbone', 'backbone.bottom_up'],
               'Misc_-_cascade_mask_rcnn_R_50_FPN_1x.yaml': ['backbone', 'backbone.bottom_up'],
               'Misc_-_cascade_mask_rcnn_R_50_FPN_3x.yaml': ['backbone', 'backbone.bottom_up'],
               'Misc_-_mask_rcnn_R_50_FPN_3x_syncbn.yaml': ['backbone', 'backbone.bottom_up'],
               'Misc_-_mask_rcnn_R_50_FPN_3x_gn.yaml': ['backbone', 'backbone.bottom_up'],
               'Misc_-_scratch_mask_rcnn_R_50_FPN_3x_gn.yaml': ['backbone', 'backbone.bottom_up'],
               'Misc_-_scratch_mask_rcnn_R_50_FPN_9x_gn.yaml': ['backbone', 'backbone.bottom_up'],
               'Misc_-_scratch_mask_rcnn_R_50_FPN_9x_syncbn.yaml': ['backbone', 'backbone.bottom_up'],
               'Misc_-_panoptic_fpn_R_101_dconv_cascade_gn_3x.yaml': ['backbone', 'backbone.bottom_up'],
               'Misc_-_cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml': ['backbone', 'backbone.bottom_up'],
               'Detectron1-Comparisons_-_faster_rcnn_R_50_FPN_noaug_1x.yaml': ['backbone', 'backbone.bottom_up'],
               'Detectron1-Comparisons_-_mask_rcnn_R_50_FPN_noaug_1x.yaml': ['backbone', 'backbone.bottom_up'],
               'Detectron1-Comparisons_-_keypoint_rcnn_R_50_FPN_1x.yaml': ['backbone', 'backbone.bottom_up']}
         




def configurator(model_name):
    """Detectron2 builds its model through configs. This function creates the config

    Args:
        model_name (str): Name of model

    Returns:
        config-type: Detectron2 config file
    """
    
    correct_model_name = model_name.replace("_-_", "/")
    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(correct_model_name))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(correct_model_name)
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
    
    final_image = V(centre_crop(image).unsqueeze(0))[0]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == torch.device('cuda'):  # send to cuda
        final_image = final_image.cuda()
    
    image_dict = [{'image': final_image}]

    return image_dict


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
    
    final_image = V(centre_crop(pil_image).unsqueeze(0))[0]
    
    # Add to Cuda    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == torch.device('cuda'):  # send to cuda
            final_image = final_image.cuda()
    
    image_dict = [{'image': final_image}]

    return image_dict
