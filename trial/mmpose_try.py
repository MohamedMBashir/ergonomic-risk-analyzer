from mmpose.apis import MMPoseInferencer
import torch

# Force CPU usage
torch.set_default_device('cpu')

input_image = "./inputs/input_old_man.jpeg"


# Initialize the inferencer

inferencer =  MMPoseInferencer(
    pose3d="human3d",
    device='cpu',
)
# or
# inferencer = MMPoseInferencer(
#     # 2D pose estimation model (RTMPose)
#     pose2d='configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb256-420e_body8-256x192.py',
#     pose2d_weights='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth',
    
#     # 3D pose estimation model (MotionBERT)
#     pose3d='configs/body_3d_keypoint/motionbert/h36m/motionbert_dstformer-ft-243frm_8xb32-120e_h36m.py',
#     pose3d_weights='https://download.openmmlab.com/mmpose/v1/body_3d_keypoint/pose_lift/h36m/motionbert_ft_h36m-d80af323_20230531.pth'
# )

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