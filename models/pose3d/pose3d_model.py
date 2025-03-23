from typing import Dict, Any
import subprocess
import os
import json
import tempfile
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
        self.vis_out_dir = vis_out_dir
        self.pred_out_dir = pred_out_dir
        self.angle_calculator = AngleCalculator()

        # Ensure the output directories exist
        os.makedirs(self.vis_out_dir, exist_ok=True)
        os.makedirs(self.pred_out_dir, exist_ok=True)

    def process_image(self, image_path: str) -> Dict[str, Any]:
        """Process image and return joint angles."""
        # Get the path to the wrapper script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        wrapper_script = os.path.join(current_dir, "mmpose_wrapper.py")

        # Create a temporary file for the results
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
            temp_file_path = temp_file.name

        try:
            # Create the conda run command to execute the wrapper in the openmmlab environment
            cmd_str = f'conda run -n openmmlab python "{wrapper_script}" "{image_path}" "{self.vis_out_dir}" "{self.pred_out_dir}" "{temp_file_path}"'

            # Run the command
            process = subprocess.run(
                cmd_str, text=True, capture_output=True, check=False, shell=True
            )

            if process.returncode != 0:
                print(f"Process stderr: {process.stderr}")
                print(f"Process stdout: {process.stdout}")
                raise RuntimeError(
                    f"MMPose wrapper failed with code {process.returncode}"
                )

            # Read the results from the temporary file
            with open(temp_file_path, "r") as f:
                result_json = f.read()

            # Parse the JSON
            result = json.loads(result_json)

            # Check for errors
            if "error" in result:
                raise RuntimeError(f"Error in MMPose inference: {result['error']}")

            # Convert lists back to numpy arrays if needed
            for key in result:
                if isinstance(result[key], list):
                    for i, item in enumerate(result[key]):
                        if isinstance(item, dict):
                            for k, v in item.items():
                                if isinstance(v, list) and k != "keypoint_scores":
                                    item[k] = np.array(v)

            # Calculate angles and adjustments
            angles, adjustments = self.angle_calculator.calculate_angles(result)

            return {"angles": angles, "adjustments": adjustments}

        except Exception as e:
            raise RuntimeError(f"Error processing image: {str(e)}")

        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except:
                    pass

    def get_model_type(self) -> str:
        """Return model type."""
        return "angle-based"
