from typing import Dict, Any
from mmpose.apis import MMPoseInferencer
import numpy as np
from ..base_model import BaseErgonomicModel
from .angle_calculator import AngleCalculator


class Pose3DModel(BaseErgonomicModel):
    """3D Pose estimation based ergonomic assessment model."""

    def __init__(
        self, vis_out_dir: str = "./outputs/vis", pred_out_dir: str = "./outputs/pred"
    ):
        """Initialize the 3D Pose estimation model.

        Args:
            vis_out_dir (str): Directory for visualization outputs
            pred_out_dir (str): Directory for prediction outputs
        """
        self.inferencer = MMPoseInferencer(
            pose3d="human3d",
            device="cpu",  # Default to CPU
        )
        self.vis_out_dir = vis_out_dir
        self.pred_out_dir = pred_out_dir
        self.angle_calculator = AngleCalculator()

    def process_image(self, image_path: str) -> Dict[str, Any]:
        """Process image and return joint angles."""
        # Run inference
        result_generator = self.inferencer(
            image_path,
            show=False,
            vis_out_dir=self.vis_out_dir,
            pred_out_dir=self.pred_out_dir,
        )

        # Get first result
        result = next(result_generator)

        # Calculate angles and adjustments
        angles, adjustments = self.angle_calculator.calculate_angles(result)

        return {"angles": angles, "adjustments": adjustments}

    def get_model_type(self) -> str:
        """Return model type."""
        return "angle-based"
