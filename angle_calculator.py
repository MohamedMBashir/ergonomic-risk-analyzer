import numpy as np

keypoints_names = [
    'bottom_torso',  # 0
    'left_hip',      # 1
    'left_knee',     # 2
    'left_foot',     # 3
    'right_hip',     # 4
    'right_knee',    # 5
    'right_foot',    # 6
    'center_torso',  # 7
    'upper_torso',   # 8
    'neck_base',     # 9
    'center_head',   # 10
    'right_shoulder',# 11
    'right_elbow',   # 12
    'right_hand',    # 13
    'left_shoulder', # 14
    'left_elbow',    # 15
    'left_hand'      # 16
]

skeleton_connections = [
    # Torso
    (0, 7),  # bottom_torso to center_torso
    (7, 8),  # center_torso to upper_torso
    (8, 9),  # upper_torso to neck_base
    (9, 10), # neck_base to center_head
    
    # Left leg
    (0, 1),  # bottom_torso to left_hip
    (1, 2),  # left_hip to left_knee
    (2, 3),  # left_knee to left_foot
    
    # Right leg
    (0, 4),  # bottom_torso to right_hip
    (4, 5),  # right_hip to right_knee
    (5, 6),  # right_knee to right_foot
    
    # Left arm
    (8, 14), # upper_torso to left_shoulder
    (14, 15),# left_shoulder to left_elbow
    (15, 16),# left_elbow to left_hand
    
    # Right arm
    (8, 11), # upper_torso to right_shoulder
    (11, 12),# right_shoulder to right_elbow
    (12, 13), # right_elbow to right_hand
    
]


class AngleCalculator:
    def __init__(self):
        pass

    def calculate_angles(self, keypoints_dict):
        # 1.Get keypoints
        keypoints = np.array(keypoints_dict['predictions'][0][0]['keypoints'])

        # 2. Rotate keypoints
        rotated_keypoints = self.rotate_keypoints(keypoints=keypoints, view="side")

        # 3. Calculate angles
        angles = {
            'upper_arm': self.get_upper_arm_angle(rotated_keypoints),
            'lower_arm': self.get_lower_arm_angle(rotated_keypoints),
            'wrist': self.get_wrist_angle(rotated_keypoints),
            'wrist_twist': self.get_wrist_twist_angle(rotated_keypoints),
            'neck': self.get_neck_angle(rotated_keypoints),
            'trunk': self.get_trunk_angle(rotated_keypoints),
            'leg': self.get_leg_angle(rotated_keypoints)
        }

        return angles
    
    def rotate_keypoints(self, keypoints, view="side"):
        right_shoulder_coords = keypoints[keypoints_names.index('right_shoulder')].copy()
        left_shoulder_coords = keypoints[keypoints_names.index('left_shoulder')].copy()
        lower_torso_coords = keypoints[keypoints_names.index('bottom_torso')].copy()
        upper_torso_coords = keypoints[keypoints_names.index('upper_torso')].copy()


        # Step 1: Calculate torso and shoulder vectors
        torso_vector = upper_torso_coords - lower_torso_coords
        shoulder_vector = right_shoulder_coords - left_shoulder_coords

        # Step 2: Compute facing vector (cross product)
        facing_vector = np.cross(shoulder_vector, torso_vector)

        # Step 3: Normalize facing vector
        facing_vector = facing_vector / np.linalg.norm(facing_vector)

        # Step 4: Define reference direction 
        if view == "side":
            #(Y-axis)
            reference_vector = np.array([0, 1, 0])
        elif view == "top":
            #(Z-axis)
            reference_vector = np.array([0, 0, 1])
        else:
            #(Y-axis)
            reference_vector = np.array([0, 1, 0])

        # Step 5: Compute angle between facing_vector and reference_vector
        dot_product = np.dot(facing_vector, reference_vector)
        cross_product = np.cross(facing_vector, reference_vector)
        angle = np.arctan2(np.linalg.norm(cross_product), dot_product)

        # Step 6: Build rotation matrix to rotate about Z-axis
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [0,              0,             1]
        ])

        # Step 7: Rotate keypoints
        rotated_keypoints = np.dot(keypoints, rotation_matrix.T)

        rotated_keypoints = rotated_keypoints * [1, -1, 1]

        return rotated_keypoints
    
    def calculate_angle_between(self, keypoint_1, keypoint_center, keypoint_2):
        vector_1 = keypoint_1 - keypoint_center
        vector_2 = keypoint_2 - keypoint_center
        
        # calculate the angle using arctan for robustness
        angle = np.arctan2(vector_1[1], vector_1[0]) - np.arctan2(vector_2[1], vector_2[0])
        
        # calculate the angle using arccos just to try it out
        # angle = np.arccos(np.dot(vector_1, vector_2) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2)))
        
        # convert the angle to degrees
        angle = np.degrees(angle)
        
        return angle

 
    def get_upper_arm_angle(self, keypoints):
        pass

    def get_lower_arm_angle(self, keypoints):
        pass

    def get_wrist_angle(self, keypoints):
        pass

    def get_wrist_twist_angle(self, keypoints):
        pass

    def get_neck_angle(self, keypoints):
        pass

    def get_trunk_angle(self, keypoints):
        pass

    def get_leg_angle(self, keypoints):
        pass


        
