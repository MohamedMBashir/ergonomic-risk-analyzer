import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

keypoints_names = [
    "bottom_torso",  # 0
    "left_hip",  # 1
    "left_knee",  # 2
    "left_foot",  # 3
    "right_hip",  # 4
    "right_knee",  # 5
    "right_foot",  # 6
    "center_torso",  # 7
    "upper_torso",  # 8
    "neck_base",  # 9
    "center_head",  # 10
    "right_shoulder",  # 11
    "right_elbow",  # 12
    "right_hand",  # 13
    "left_shoulder",  # 14
    "left_elbow",  # 15
    "left_hand",  # 16
]

skeleton_connections = [
    # Torso
    (0, 7),  # bottom_torso to center_torso
    (7, 8),  # center_torso to upper_torso
    (8, 9),  # upper_torso to neck_base
    (9, 10),  # neck_base to center_head
    # Left leg
    (0, 1),  # bottom_torso to left_hip
    (1, 2),  # left_hip to left_knee
    (2, 3),  # left_knee to left_foot
    # Right leg
    (0, 4),  # bottom_torso to right_hip
    (4, 5),  # right_hip to right_knee
    (5, 6),  # right_knee to right_foot
    # Left arm
    (8, 14),  # upper_torso to left_shoulder
    (14, 15),  # left_shoulder to left_elbow
    (15, 16),  # left_elbow to left_hand
    # Right arm
    (8, 11),  # upper_torso to right_shoulder
    (11, 12),  # right_shoulder to right_elbow
    (12, 13),  # right_elbow to right_hand
]


class AngleCalculator:
    def __init__(self):
        pass

    def calculate_angles(self, keypoints_dict):
        # 1.Get keypoints
        keypoints = np.array(keypoints_dict["predictions"][0][0]["keypoints"])

        # 2. Rotate keypoints for different views
        side_keypoints = self.rotate_keypoints(keypoints=keypoints, view="side")
        front_keypoints = self.rotate_keypoints(keypoints=keypoints, view="front")
        side_keypoints = side_keypoints[:, 1:]

        # Calculate shoulder raised angle from front view
        shoulder_angle = self._get_shoulder_angle(front_keypoints)
        shoulder_raised = 1 if abs(shoulder_angle - 180) > 10 else 0

        # Calculate arm abduction from front view
        arm_abduction_angle = self._get_arm_abduction_angle(front_keypoints)
        upper_arm_abducted = 1 if arm_abduction_angle > 90 else 0

        # Calculate neck bending from front view
        neck_bend_angle = self._get_neck_bend_angle(front_keypoints)
        neck_bended = 1 if abs(neck_bend_angle) > 10 else 0

        # Calculate trunk bending from front view
        trunk_bend_angle = self._get_trunk_bend_angle(front_keypoints)
        trunk_bended = 1 if abs(trunk_bend_angle) > 10 else 0

        # Calculate arm position from top view
        top_keypoints = self.rotate_keypoints(keypoints, view="top")
        arms_outside = self._get_arms_outside_inside(top_keypoints)

        # 3. Calculate angles and adjustments
        angles = {
            "upper_arm": self.get_upper_arm_angle(side_keypoints),
            "lower_arm": self.get_lower_arm_angle(side_keypoints),
            "wrist": self.get_wrist_angle(side_keypoints),
            "wrist_twist": self.get_wrist_twist_angle(side_keypoints),
            "neck": self.get_neck_angle(side_keypoints),
            "trunk": self.get_trunk_angle(side_keypoints),
            "leg": self.get_leg_angle(side_keypoints),
        }

        adjustments = {
            "shoulder_raised": shoulder_raised,
            "upper_arm_abducted": upper_arm_abducted,
            "arm_supported_or_leaning": 0,
            "arms_outside_inside": arms_outside,
            "wrist_bend": 0,
            "neck_twist": 0,
            "neck_bended": neck_bended,
            "trunk_twisted": 0,
            "trunk_bended": trunk_bended,
        }

        return angles, adjustments

    def rotate_keypoints(self, keypoints, view="side"):
        right_shoulder_coords = keypoints[
            keypoints_names.index("right_shoulder")
        ].copy()
        left_shoulder_coords = keypoints[keypoints_names.index("left_shoulder")].copy()
        lower_torso_coords = keypoints[keypoints_names.index("bottom_torso")].copy()
        upper_torso_coords = keypoints[keypoints_names.index("upper_torso")].copy()

        if view == "top":
            # Step 1: Compute torso vector
            torso_vector = upper_torso_coords - lower_torso_coords
            torso_vector = torso_vector / np.linalg.norm(torso_vector)

            # Step 2: Define target up direction
            target_vector = np.array([0, 0, 1])

            # Step 3: Compute rotation axis and angle
            rotation_axis = np.cross(torso_vector, target_vector)
            if np.linalg.norm(rotation_axis) > 0:
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

                dot_product = np.dot(torso_vector, target_vector)
                angle = np.arccos(np.clip(dot_product, -1.0, 1.0))

                # Step 4: Compute rotation matrix using Rodrigues' formula
                K = np.array(
                    [
                        [0, -rotation_axis[2], rotation_axis[1]],
                        [rotation_axis[2], 0, -rotation_axis[0]],
                        [-rotation_axis[1], rotation_axis[0], 0],
                    ]
                )

                I = np.eye(3)
                rotation_matrix = (
                    I + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
                )

                # Step 5: Rotate keypoints
                rotated_keypoints = np.dot(keypoints, rotation_matrix.T)

                # Step 6: Flip Z-axis for consistent top-down view
                rotated_keypoints = rotated_keypoints * [1, 1, -1]

            else:
                rotated_keypoints = keypoints.copy() * [1, 1, -1]

        elif view == "front":
            # Step 1: Compute the torso vector (main body direction)
            torso_vector = upper_torso_coords - lower_torso_coords
            torso_vector = torso_vector / np.linalg.norm(torso_vector)

            # Step 2: Define the target forward direction (Z-axis)
            target_vector = np.array([0, 0, 1])

            # Step 3: Compute rotation axis and angle
            rotation_axis = np.cross(torso_vector, target_vector)
            if np.linalg.norm(rotation_axis) > 0:  # Check if vectors aren't parallel
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

                dot_product = np.dot(torso_vector, target_vector)
                angle = np.arccos(np.clip(dot_product, -1.0, 1.0))

                # Step 4: Compute rotation matrix using Rodrigues' formula
                K = np.array(
                    [
                        [0, -rotation_axis[2], rotation_axis[1]],
                        [rotation_axis[2], 0, -rotation_axis[0]],
                        [-rotation_axis[1], rotation_axis[0], 0],
                    ]
                )

                I = np.eye(3)
                rotation_matrix = (
                    I + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
                )

                # Step 5: Rotate keypoints
                rotated_keypoints = np.dot(keypoints, rotation_matrix.T)

                # Step 6: Flip X-axis for correct left-right orientation
                rotated_keypoints = rotated_keypoints * [-1, 1, 1]

        else:  # "side" view (existing code)
            # Step 1: Calculate torso and shoulder vectors
            torso_vector = upper_torso_coords - lower_torso_coords
            shoulder_vector = right_shoulder_coords - left_shoulder_coords

            # Step 2: Compute facing vector (cross product)
            facing_vector = np.cross(shoulder_vector, torso_vector)

            # Step 3: Normalize facing vector
            facing_vector = facing_vector / np.linalg.norm(facing_vector)

            # Step 4: Define reference direction (Y-axis for side view)
            reference_vector = np.array([0, 1, 0])

            # Step 5: Compute angle between facing_vector and reference_vector
            dot_product = np.dot(facing_vector, reference_vector)
            cross_product = np.cross(facing_vector, reference_vector)
            angle = np.arctan2(np.linalg.norm(cross_product), dot_product)

            # Step 6: Build rotation matrix to rotate about Z-axis
            rotation_matrix = np.array(
                [
                    [np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1],
                ]
            )

            # Step 7: Rotate keypoints
            rotated_keypoints = np.dot(keypoints, rotation_matrix.T)
            rotated_keypoints = rotated_keypoints * [1, -1, 1]

        return rotated_keypoints

    def calculate_angle_between(self, keypoint_1, keypoint_center, keypoint_2):
        vector_1 = keypoint_1 - keypoint_center
        vector_2 = keypoint_2 - keypoint_center

        # calculate the angle using arctan for robustness
        # angle = np.arctan2(vector_1[1], vector_1[0]) - np.arctan2(vector_2[1], vector_2[0])

        # calculate the angle using arccos just to try it out
        angle = np.arccos(
            np.dot(vector_1, vector_2)
            / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2))
        )

        # convert the angle to degrees
        angle = np.degrees(angle)

        return angle

    # ✅ TODO: Do Adjustments
    def get_upper_arm_angle(self, keypoints):
        point1 = keypoints[keypoints_names.index("right_elbow")]
        point2 = keypoints[keypoints_names.index("right_shoulder")]
        point3 = keypoints[keypoints_names.index("center_torso")]
        return abs(self.calculate_angle_between(point1, point2, point3))

    # ✅ TODO: Do Adjustments
    def get_lower_arm_angle(self, keypoints):
        point1 = keypoints[keypoints_names.index("right_shoulder")]
        point2 = keypoints[keypoints_names.index("right_elbow")]
        point3 = keypoints[keypoints_names.index("right_hand")]
        return 180 - abs(self.calculate_angle_between(point1, point2, point3))

    # ❌ TODO:Need another model. TODO: Do Adjustments
    def get_wrist_angle(self, keypoints):
        # Wrist twist might require additional data points or different calculation method
        # For now, returning 0 as placeholder
        return 0

    # ❌ TODO:Need another model. TODO: DoAdjustments
    def get_wrist_twist_angle(self, keypoints):
        # Wrist twist might require additional data points or different calculation method
        # For now, returning 0 as placeholder
        return 0

    # ✅ TODO: Do Adjustments
    def get_neck_angle(self, keypoints):
        point1 = keypoints[keypoints_names.index("center_torso")]
        point2 = keypoints[keypoints_names.index("upper_torso")]
        point3 = keypoints[keypoints_names.index("center_head")]
        angle = 180 - abs(self.calculate_angle_between(point1, point2, point3))
        angle -= 15  # Correction Factor
        return angle

    # ✅ TODO: Do Adjustments
    # NOTE: We must replace right_foot with floor position.
    def get_trunk_angle(self, keypoints):
        point1 = keypoints[keypoints_names.index("upper_torso")]
        point2 = keypoints[keypoints_names.index("bottom_torso")]
        point3 = keypoints[keypoints_names.index("right_foot")]
        return 180 - abs(self.calculate_angle_between(point1, point2, point3))

    # ✅ Correct
    def get_leg_angle(self, keypoints):
        point1 = keypoints[keypoints_names.index("right_foot")]
        point2 = keypoints[keypoints_names.index("bottom_torso")]
        point3 = keypoints[keypoints_names.index("left_foot")]
        return abs(self.calculate_angle_between(point1, point2, point3))

    def _get_shoulder_angle(self, keypoints):
        """
        Calculate the angle between left shoulder, neck base, and right shoulder
        to determine if shoulders are raised.
        """
        left_shoulder = keypoints[keypoints_names.index("left_shoulder")]
        right_shoulder = keypoints[keypoints_names.index("right_shoulder")]
        neck_base = keypoints[keypoints_names.index("neck_base")]

        return self.calculate_angle_between(left_shoulder, neck_base, right_shoulder)

    def _get_neck_bend_angle(self, keypoints):
        """
        Calculate the neck bend angle in the front view.
        Measures deviation from vertical between upper_torso and head.

        Returns:
            float: Angle in degrees. Positive means bent to right, negative to left.
        """
        upper_torso = keypoints[keypoints_names.index("upper_torso")]
        neck_base = keypoints[keypoints_names.index("neck_base")]
        head = keypoints[keypoints_names.index("center_head")]

        # Create vertical reference point above neck_base
        vertical_point = neck_base.copy()
        vertical_point[2] += 1.0  # Move up in Z direction

        # Calculate angle between actual neck vector and vertical vector
        neck_angle = self.calculate_angle_between(head, neck_base, vertical_point)

        # Determine direction of bend (left/right) using cross product
        neck_vector = head - neck_base
        vertical_vector = vertical_point - neck_base
        cross_product = np.cross(vertical_vector, neck_vector)

        # If cross product's Y component is negative, bend is to the left
        return -neck_angle if cross_product[1] < 0 else neck_angle

    def _get_trunk_bend_angle(self, keypoints):
        """
        Calculate the trunk bend angle in the front view.
        Measures deviation from vertical between bottom_torso and upper_torso.

        Returns:
            float: Angle in degrees. Positive means bent to right, negative to left.
        """
        bottom_torso = keypoints[keypoints_names.index("bottom_torso")]
        upper_torso = keypoints[keypoints_names.index("upper_torso")]

        # Create vertical reference point above bottom_torso
        vertical_point = bottom_torso.copy()
        vertical_point[2] += 1.0  # Move up in Z direction

        # Calculate angle between actual trunk vector and vertical vector
        trunk_angle = self.calculate_angle_between(
            upper_torso, bottom_torso, vertical_point
        )

        # Determine direction of bend using cross product
        trunk_vector = upper_torso - bottom_torso
        vertical_vector = vertical_point - bottom_torso
        cross_product = np.cross(vertical_vector, trunk_vector)

        # If cross product's Y component is negative, bend is to the left
        return -trunk_angle if cross_product[1] < 0 else trunk_angle

    def _get_arm_abduction_angle(self, keypoints):
        """
        Calculate the arm abduction angle in the front view.
        Measures angle between right elbow, right shoulder and upper torso.

        Returns:
            float: Angle in degrees. > 180 means arm is abducted (moved away from body)
        """
        right_elbow = keypoints[keypoints_names.index("right_elbow")]
        right_shoulder = keypoints[keypoints_names.index("right_shoulder")]
        upper_torso = keypoints[keypoints_names.index("upper_torso")]

        # Calculate angle between elbow-shoulder-torso
        abduction_angle = self.calculate_angle_between(
            right_elbow, right_shoulder, upper_torso
        )

        return abduction_angle

    def _get_arms_outside_inside(self, keypoints):
        """
        Calculate if arms are positioned outside or inside using top view angles.
        Uses two angles:
        1. Right shoulder-elbow-hand angle (arm bend)
        2. Left shoulder-right shoulder-right elbow angle (arm position relative to shoulders)

        Returns:
            int: 1 if arms are outside working envelope, 0 if inside
        """
        right_hand = keypoints[keypoints_names.index("right_hand")]
        right_elbow = keypoints[keypoints_names.index("right_elbow")]
        right_shoulder = keypoints[keypoints_names.index("right_shoulder")]
        left_shoulder = keypoints[keypoints_names.index("left_shoulder")]

        # Calculate arm bend angle
        arm_bend_angle = self.calculate_angle_between(
            right_hand, right_elbow, right_shoulder
        )

        # Calculate arm position relative to shoulders
        shoulder_arm_angle = self.calculate_angle_between(
            left_shoulder, right_shoulder, right_elbow
        )

        # Combined angle (subtract from 180 to get deviation from straight line)
        combined_angle = 180 - (arm_bend_angle + shoulder_arm_angle)

        # If combined angle > 90, arms are significantly outside the working envelope
        return 1 if abs(combined_angle) > 90 else 0
