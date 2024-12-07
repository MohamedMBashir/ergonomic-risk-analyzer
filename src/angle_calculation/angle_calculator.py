import numpy as np
from typing import Dict, List

class AngleCalculator:
    def calculate_angles(self, pose_data) -> Dict[str, float]:
        """Calculate relevant joint angles for ergonomic assessment"""
        angles = {
            'neck': self._calculate_neck_angle(pose_data),
            'trunk': self._calculate_trunk_angle(pose_data),
            'upper_arm_right': self._calculate_upper_arm_angle(pose_data, side='right'),
            'upper_arm_left': self._calculate_upper_arm_angle(pose_data, side='left'),
            'lower_arm_right': self._calculate_lower_arm_angle(pose_data, side='right'),
            'lower_arm_left': self._calculate_lower_arm_angle(pose_data, side='left'),
            # Add more angles as needed
        }
        return angles
    
    def _calculate_angle_3d(self, point1, point2, point3) -> float:
        """Calculate angle between three 3D points"""
        vector1 = point1 - point2
        vector2 = point3 - point2
        
        cos_angle = np.dot(vector1, vector2) / (
            np.linalg.norm(vector1) * np.linalg.norm(vector2)
        )
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return np.degrees(angle)
    
    def _calculate_neck_angle(self, pose_data):
        """Calculate neck flexion/extension angle"""
        # Implementation specific to your pose format
        pass
    
    # Add more angle calculation methods