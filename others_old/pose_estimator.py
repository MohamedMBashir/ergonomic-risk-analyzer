from mmpose.apis import MMPoseInferencer
import numpy as np


class PoseEstimator:
    def __init__(self, vis_out_dir, pred_out_dir):
        """Initialize the pose estimator with default configuration"""
        self.inferencer = MMPoseInferencer(
            pose3d="human3d",
            device="cpu",  # Default to CPU
        )./
        self.vis_out_dir = vis_out_dir if vis_out_dir else "./outputs/vis"
        self.pred_out_dir = pred_out_dir if pred_out_dir else "./outputs/pred"

    def estimate_pose(self, input_image, show=False):
        """
        Estimate 3D pose from input image

        Args:
            input_image (str): Path to input image
            show (bool): Whether to show visualization

        Returns:
            dict: Dictionary containing keypoints and visualization data
        """
        # Run inference
        result_generator = self.inferencer(
            input_image,
            show=show,
            vis_out_dir=self.vis_out_dir,
            pred_out_dir=self.pred_out_dir,
        )

        # Get first result (for single image)
        result = next(result_generator)

        return result
