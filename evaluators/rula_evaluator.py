from typing import Dict, Any
import numpy as np
from models.base_model import BaseErgonomicModel


class RULAEvaluator:
    """RULA score calculator that works with both angle-based and score-based models."""

    def __init__(self):
        """Initialize lookup tables for RULA calculation."""
        self.table_a = np.array(
            [
                [
                    [[1, 2], [2, 2], [2, 3], [3, 3]],
                    [[2, 2], [2, 2], [3, 3], [3, 3]],
                    [[2, 3], [3, 3], [3, 3], [4, 4]],
                ],
                [
                    [[2, 3], [3, 3], [3, 4], [4, 4]],
                    [[3, 3], [3, 3], [3, 4], [4, 4]],
                    [[3, 4], [4, 4], [4, 4], [5, 5]],
                ],
                [
                    [[3, 3], [4, 4], [4, 4], [5, 5]],
                    [[3, 4], [4, 4], [4, 4], [5, 5]],
                    [[4, 4], [4, 4], [4, 5], [5, 5]],
                ],
                [
                    [[4, 4], [4, 4], [4, 5], [5, 5]],
                    [[4, 4], [4, 4], [4, 5], [5, 5]],
                    [[4, 4], [4, 5], [5, 5], [6, 6]],
                ],
                [
                    [[5, 5], [5, 5], [5, 6], [6, 7]],
                    [[5, 6], [6, 6], [6, 7], [7, 7]],
                    [[6, 6], [6, 7], [7, 7], [7, 8]],
                ],
                [
                    [[7, 7], [7, 7], [7, 8], [8, 9]],
                    [[8, 8], [8, 8], [8, 9], [9, 9]],
                    [[9, 9], [9, 9], [9, 9], [9, 9]],
                ],
            ]
        )

        self.table_b = np.array(
            [
                [[1, 3], [2, 3], [3, 4], [5, 5], [6, 6], [7, 7]],
                [[2, 3], [2, 3], [4, 5], [5, 5], [6, 7], [7, 7]],
                [[3, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 7]],
                [[5, 5], [5, 6], [6, 7], [7, 7], [7, 7], [8, 8]],
                [[7, 7], [7, 7], [7, 8], [8, 8], [8, 8], [8, 8]],
                [[8, 8], [8, 8], [8, 8], [8, 9], [9, 9], [9, 9]],
            ]
        )

        self.table_c = np.array(
            [
                [1, 2, 3, 3, 4, 5, 5],
                [2, 2, 3, 4, 4, 5, 5],
                [3, 3, 3, 4, 4, 5, 6],
                [3, 3, 3, 4, 5, 6, 6],
                [4, 4, 4, 5, 6, 7, 7],
                [4, 4, 5, 6, 6, 7, 7],
                [5, 5, 6, 6, 7, 7, 7],
                [5, 5, 6, 7, 7, 7, 7],
            ]
        )

    def _calc_upper_arm_score(self, angle: float, adjustments: Dict) -> int:
        """Calculate upper arm score from angle."""
        if 20 >= angle >= -20:
            score = 1
        elif -20 > angle:
            score = 2
        elif 45 >= angle > 20:
            score = 2
        elif 90 >= angle > 45:
            score = 3
        elif 90 < angle:
            score = 4

        if adjustments.get("shoulder_raised"):
            score += 1
        if adjustments.get("upper_arm_abducted"):
            score += 1
        if adjustments.get("arm_supported_or_leaning"):
            score -= 1

        return max(1, min(score, 6))

    def _calc_lower_arm_score(self, angle: float, adjustments: Dict) -> int:
        """Calculate lower arm score from angle."""
        if 100 >= angle >= 60:
            score = 1
        else:
            score = 2

        if adjustments.get("arms_outside_inside"):
            score += 1

        return max(1, min(score, 3))

    def _calc_wrist_score(self, angle: float, adjustments: Dict) -> int:
        """Calculate wrist score from angle."""
        if 1 >= angle >= -1:
            score = 1
        elif 15 >= angle >= -15:
            score = 2
        else:
            score = 3

        if adjustments.get("wrist_bend", 0) > 10:
            score += 1

        return max(1, min(score, 4))

    def _calc_neck_score(self, angle: float, adjustments: Dict) -> int:
        """Calculate neck score from angle."""
        if 0 <= angle <= 10:
            score = 1
        elif 10 < angle <= 20:
            score = 2
        elif 20 < angle:
            score = 3
        else:
            score = 4

        if adjustments.get("neck_twist"):
            score += 1
        if adjustments.get("neck_bended"):
            score += 1

        return max(1, min(score, 6))

    def _calc_trunk_score(self, angle: float, adjustments: Dict) -> int:
        """Calculate trunk score from angle."""
        if 0 <= angle <= 2:
            score = 1
        elif 2 < angle <= 20:
            score = 2
        elif 20 < angle <= 60:
            score = 3
        else:
            score = 4

        if adjustments.get("trunk_twisted"):
            score += 1
        if adjustments.get("trunk_bended"):
            score += 1

        return max(1, min(score, 6))

    def _calc_leg_score(self, angle: float, adjustments: Dict) -> int:
        """Calculate leg score from angle."""
        return 2 if angle >= 10 else 1

    def evaluate(
        self, model_output: Dict[str, Any], force_load: int = 1, muscle_use: int = 1
    ) -> Dict[str, Any]:
        """Calculate RULA score from model output.

        Args:
            model_output: Output from a model's process_image method
            force_load: Force/load score (default: 1)
            muscle_use: Muscle use score (default: 1)

        Returns:
            Dict containing final RULA score and component scores
        """
        if "angles" in model_output:  # Angle-based model
            angles = model_output["angles"]
            adjustments = model_output.get("adjustments", {})

            # Calculate component scores from angles
            scores = {
                "upper_arm_score": self._calc_upper_arm_score(
                    angles["upper_arm"], adjustments
                ),
                "lower_arm_score": self._calc_lower_arm_score(
                    angles["lower_arm"], adjustments
                ),
                "wrist_score": self._calc_wrist_score(angles["wrist"], adjustments),
                "wrist_twist_score": 1,  # Default as it's hard to determine
                "neck_score": self._calc_neck_score(angles["neck"], adjustments),
                "trunk_score": self._calc_trunk_score(angles["trunk"], adjustments),
                "leg_score": self._calc_leg_score(angles["leg"], adjustments),
            }
        else:  # Score-based model
            scores = model_output["scores"]

        # Calculate table scores
        table_a_score = self.table_a[
            scores["upper_arm_score"] - 1,
            scores["lower_arm_score"] - 1,
            scores["wrist_score"] - 1,
            scores["wrist_twist_score"] - 1,
        ]

        table_b_score = self.table_b[
            scores["neck_score"] - 1, scores["trunk_score"] - 1, scores["leg_score"] - 1
        ]

        # Add force/load and muscle use scores
        score_a = table_a_score + force_load + muscle_use
        score_b = table_b_score + force_load + muscle_use

        # Calculate final score
        if score_a > 8:
            score_a = 8
        if score_b > 7:
            score_b = 7

        final_score = self.table_c[score_a - 1, score_b - 1]

        return {
            "final_score": final_score,
            "score_a": score_a,
            "score_b": score_b,
            "table_a_score": table_a_score,
            "table_b_score": table_b_score,
            **scores,  # Include component scores
        }
