import sys
import os
import os.path as osp
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from models.mppe.mppe_repo.main.config import cfg
from models.mppe.mppe_repo.main.model import get_pose_net
from models.mppe.mppe_repo.common.utils.pose_utils import process_bbox

# When this module is imported, keep track of original path
_original_sys_path = list(sys.path)


def get_root_depth(img_path, bbox_list):
    # Restore original path when function is called
    sys.path = list(_original_sys_path)

    # Prepare input image
    original_img = cv2.imread(img_path)
    original_img_height, original_img_width = original_img.shape[:2]

    # Create transform
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std),
        ]
    )

    # Get model and load demo model weights (simplified example)
    model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "models/snapshot_18.pth.tar"
    )
    assert os.path.exists(model_path), f"Model path {model_path} does not exist"

    # Load root depth model
    # This is a simplified example - you would need to adapt this to your actual root depth detection logic
    # For now, returning fixed depth values for demonstration
    root_depth_list = [500.0] * len(bbox_list)  # Example fixed depth values

    return root_depth_list
