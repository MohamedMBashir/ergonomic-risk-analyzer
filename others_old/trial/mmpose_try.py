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
    show=False,                 # Show visualization
    vis_out_dir='./outputs/vis',  # Save visualization results
    pred_out_dir='./outputs/pred' # Save prediction results
)

# Get results
result = next(result_generator)

print(result)

