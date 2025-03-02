from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple


class BaseErgonomicModel(ABC):
    """Base class for all ergonomic assessment models."""

    @abstractmethod
    def process_image(self, image_path: str) -> Dict[str, Any]:
        """Process an image and return either angles or direct RULA scores.

        Args:
            image_path (str): Path to the input image

        Returns:
            Dict containing either:
            - angles: Dict of joint angles in degrees
            - scores: Dict of direct RULA scores for each component
            - adjustments: Dict of any adjustments needed for RULA calculation
        """
        pass

    @abstractmethod
    def get_model_type(self) -> str:
        """Return the type of model (angle-based or score-based)."""
        pass
