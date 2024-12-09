from mmpose.apis import MMPoseInferencer
import torch

# Force CPU usage
torch.set_default_device('cpu')

# input_image = "./inputs/input_old_man.jpeg"
input_image = "./inputs/rula_test_girl.jpg"

# Initialize the inferencer

inferencer =  MMPoseInferencer(
    pose3d="human3d",
    device='cpu',
)

# Run inference
result_generator = inferencer(
    input_image,  # Can be image, video, or webcam
    # show=True,                 # Show visualization
    # radius=4,          # Keypoint visualization radius
    # thickness=2,       # Skeleton line thickness
    # draw_heatmap=False, # Show heatmaps
    # show_interval=1,    # Frame display interval for videos
    vis_out_dir='./outputs/vis',  # Save visualization results
    pred_out_dir='./outputs/pred' # Save prediction results
)

# Get results
result = next(result_generator)

print(result)


# keypoints_names = [
#     'bottom_torso',  # 0
#     'left_hip',      # 1
#     'left_knee',     # 2
#     'left_foot',     # 3
#     'right_hip',     # 4
#     'right_knee',    # 5
#     'right_foot',    # 6
#     'center_torso',  # 7
#     'upper_torso',   # 8
#     'neck_base',     # 9
#     'center_head',   # 10
#     'right_shoulder',# 11
#     'right_elbow',   # 12
#     'right_hand',    # 13
#     'left_shoulder', # 14
#     'left_elbow',    # 15
#     'left_hand'      # 16
# ]