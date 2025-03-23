import os
import sys
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from typing import Dict, Any

# Get the absolute path to the project root and mppe_repo directory
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.join(current_dir, "mppe_repo")
project_root = os.path.abspath(os.path.join(current_dir, "../.."))

# Add project root and repo_dir to the Python path
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)

# Import the required modules
from models.mppe.mppe_repo.main.config import cfg
from models.mppe.mppe_repo.main.model import get_pose_net
from models.mppe.mppe_repo.data.dataset import generate_patch_image
from models.mppe.mppe_repo.demo.root_demo import get_root_depth
from models.mppe.mppe_repo.common.utils.yolo_utils import get_bbox_list
from models.mppe.mppe_repo.common.utils.pose_utils import process_bbox, pixel2cam

from ..base_model import BaseErgonomicModel
from .angle_calculator import AngleCalculator


class Pose3DMPPEModel(BaseErgonomicModel):
    """3D Multi-Person Pose Estimation based ergonomic assessment model."""

    # Original 3D MPPE keypoints
    JOINT_NUM = 21
    JOINTS_NAME = (
        "Head_top",
        "Thorax",
        "R_Shoulder",
        "R_Elbow",
        "R_Wrist",
        "L_Shoulder",
        "L_Elbow",
        "L_Wrist",
        "R_Hip",
        "R_Knee",
        "R_Ankle",
        "L_Hip",
        "L_Knee",
        "L_Ankle",
        "Pelvis",
        "Spine",
        "Head",
        "R_Hand",
        "L_Hand",
        "R_Toe",
        "L_Toe",
    )

    # Mapping from 3D MPPE to AngleCalculator keypoints
    KEYPOINT_MAPPING = {
        "bottom_torso": "Pelvis",  # 0 -> 14
        "left_hip": "L_Hip",  # 1 -> 11
        "left_knee": "L_Knee",  # 2 -> 12
        "left_foot": "L_Ankle",  # 3 -> 13
        "right_hip": "R_Hip",  # 4 -> 8
        "right_knee": "R_Knee",  # 5 -> 9
        "right_foot": "R_Ankle",  # 6 -> 10
        "center_torso": "Spine",  # 7 -> 15
        "upper_torso": "Thorax",  # 8 -> 1
        "neck_base": "Thorax",  # 9 -> 1 (approximation)
        "center_head": "Head",  # 10 -> 16
        "right_shoulder": "R_Shoulder",  # 11 -> 2
        "right_elbow": "R_Elbow",  # 12 -> 3
        "right_hand": "R_Hand",  # 13 -> 17
        "left_shoulder": "L_Shoulder",  # 14 -> 5
        "left_elbow": "L_Elbow",  # 15 -> 6
        "left_hand": "L_Hand",  # 16 -> 18
    }

    def __init__(
        self,
        model_path: str = None,
        vis_out_dir: str = "./outputs/vis",
        pred_out_dir: str = "./outputs/pred",
        device: str = "cpu",
    ):
        """Initialize the 3D MPPE model.

        Args:
            model_path (str): Path to the model checkpoint
            vis_out_dir (str): Directory for visualization outputs
            pred_out_dir (str): Directory for prediction outputs
            device (str): Device to run inference on ('cpu' or 'cuda')
        """
        self.device = device
        self.vis_out_dir = vis_out_dir
        self.pred_out_dir = pred_out_dir

        # Create output directories if they don't exist
        os.makedirs(vis_out_dir, exist_ok=True)
        os.makedirs(pred_out_dir, exist_ok=True)

        # Initialize GPU if needed
        if device == "cuda" and torch.cuda.is_available():
            cfg.set_args("0")  # Using first GPU
        else:
            cfg.set_args("")  # Using CPU

        # Find model path if not provided
        if model_path is None:
            demo_dir = os.path.join(repo_dir, "demo")
            model_dir = os.path.join(demo_dir, "models")
            available_models = [
                f
                for f in os.listdir(model_dir)
                if f.startswith("snapshot_") and f.endswith(".pth.tar")
            ]
            if available_models:
                # Use the latest model (highest epoch number)
                latest_model = sorted(
                    available_models, key=lambda x: int(x.split("_")[1].split(".")[0])
                )[-1]
                model_path = os.path.join(model_dir, latest_model)
            else:
                raise FileNotFoundError(
                    "No model checkpoint found. Please provide a valid model_path."
                )

        # Load model
        print(f"Loading checkpoint from {model_path}")
        self.model = get_pose_net(cfg, False, self.JOINT_NUM)
        self.model = torch.nn.DataParallel(self.model)

        # Load model to specified device
        map_location = torch.device(self.device)
        ckpt = torch.load(model_path, map_location=map_location)
        self.model.load_state_dict(ckpt["network"])
        self.model.eval()

        # Initialize transform
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std),
            ]
        )

        # Initialize the angle calculator from the existing module
        self.angle_calculator = AngleCalculator()

    def process_image(self, image_path: str) -> Dict[str, Any]:
        """Process image and return joint angles.

        Args:
            image_path (str): Path to the input image

        Returns:
            Dict[str, Any]: Dictionary containing angles and adjustments
        """
        # Load image
        original_img = cv2.imread(image_path)
        if original_img is None:
            raise FileNotFoundError(f"Could not load image from {image_path}")

        original_img_height, original_img_width = original_img.shape[:2]
        focal = [1500, 1500]  # x-axis, y-axis
        princpt = [original_img_width / 2, original_img_height / 2]  # x-axis, y-axis

        # Get bounding boxes and root depth
        bbox_list = get_bbox_list(image_path)
        root_depth_list = get_root_depth(image_path, bbox_list)

        output_pose_3d_list = []

        # Process each person detected in the image
        for bbox, root_depth in zip(bbox_list, root_depth_list):
            bbox = process_bbox(np.array(bbox), original_img_width, original_img_height)
            img, img2bb_trans = generate_patch_image(
                original_img, bbox, False, 1.0, 0.0, False
            )
            img = self.transform(img).to(self.device)[None, :, :, :]

            # Run inference
            with torch.no_grad():
                pose_3d = self.model(img)[0].cpu().numpy()

            # Post-process pose
            pose_3d = self._post_process_pose(
                pose_3d, img2bb_trans, root_depth, focal, princpt
            )
            output_pose_3d_list.append(pose_3d.copy())

        # If no pose detected, return empty angles
        if not output_pose_3d_list:
            return {"angles": {}, "adjustments": {}}

        # Convert 3D MPPE keypoints to angle_calculator format
        # For simplicity, we'll use the first person's pose if multiple are detected
        mapped_keypoints = self._map_keypoints_for_angle_calculator(
            output_pose_3d_list[0]
        )

        # Create a prediction dictionary in the format expected by angle_calculator
        prediction_dict = {"predictions": [[{"keypoints": mapped_keypoints}]]}

        # Use angle_calculator to calculate angles and adjustments
        angles, adjustments = self.angle_calculator.calculate_angles(prediction_dict)

        return {"angles": angles, "adjustments": adjustments}

    def _post_process_pose(self, pose_3d, img2bb_trans, root_depth, focal, princpt):
        """Post-process the estimated 3D pose."""
        pose_3d[:, 0] = pose_3d[:, 0] / cfg.output_shape[1] * cfg.input_shape[1]
        pose_3d[:, 1] = pose_3d[:, 1] / cfg.output_shape[0] * cfg.input_shape[0]
        pose_3d_xy1 = np.concatenate((pose_3d[:, :2], np.ones_like(pose_3d[:, :1])), 1)
        img2bb_trans_001 = np.concatenate(
            (img2bb_trans, np.array([0, 0, 1]).reshape(1, 3))
        )
        pose_3d[:, :2] = np.dot(
            np.linalg.inv(img2bb_trans_001), pose_3d_xy1.transpose(1, 0)
        ).transpose(1, 0)[:, :2]
        pose_3d[:, 2] = (pose_3d[:, 2] / cfg.depth_dim * 2 - 1) * (
            cfg.bbox_3d_shape[0] / 2
        ) + root_depth
        pose_3d = pixel2cam(pose_3d, focal, princpt)
        return pose_3d

    def _map_keypoints_for_angle_calculator(self, pose_3d):
        """Map 3D MPPE keypoints to the format expected by angle_calculator."""
        # Create a dictionary to easily access 3D MPPE keypoints by name
        keypoint_dict = {name: pose_3d[i] for i, name in enumerate(self.JOINTS_NAME)}

        # Create a list with 17 keypoints in the order expected by angle_calculator
        mapped_keypoints = []

        from .angle_calculator import keypoints_names

        for name in keypoints_names:
            mppe_name = self.KEYPOINT_MAPPING.get(name)
            if mppe_name in keypoint_dict:
                # Get coordinates and reverse y-axis if needed to match angle_calculator's expectations
                coords = keypoint_dict[mppe_name].copy()
                mapped_keypoints.append(coords)
            else:
                # If no direct mapping exists, use an approximation or default position
                print(f"Warning: No mapping found for keypoint {name}")
                mapped_keypoints.append(np.zeros(3))

        return mapped_keypoints

    def get_model_type(self) -> str:
        """Return model type."""
        return "3d-mppe-angle-based"
