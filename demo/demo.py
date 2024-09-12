import sys
import os
import os.path as osp
import argparse
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from matplotlib import pyplot as plt

# Add project root and subdirectories to sys.path
def setup_environment():
    current_file_path = osp.abspath(__file__)
    project_root = osp.dirname(osp.dirname(current_file_path))
    for subdir in ['main', 'data', 'common']:
        sys.path.insert(0, osp.join(project_root, subdir))

setup_environment()

# Now we can safely import our custom modules
from config import cfg
from model import get_pose_net
from dataset import generate_patch_image
from root_demo import get_root_depth
from utils.yolo_utils import get_bbox_list
from utils.pose_utils import process_bbox, pixel2cam
from utils.vis import vis_keypoints, vis_3d_multiple_skeleton
# from risk_score import calculate_rula_score

# Constants
JOINT_NUM = 21
JOINTS_NAME = ('Head_top', 'Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Spine', 'Head', 'R_Hand', 'L_Hand', 'R_Toe', 'L_Toe')
FLIP_PAIRS = ((2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13), (17, 18), (19, 20))
SKELETON = ((0, 16), (16, 1), (1, 15), (15, 14), (14, 8), (14, 11), (8, 9), (9, 10), (10, 19), (11, 12), (12, 13), (13, 20), (1, 2), (2, 3), (3, 4), (4, 17), (1, 5), (5, 6), (6, 7), (7, 18))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids', default='')
    parser.add_argument('--test_epoch', type=str, dest='test_epoch')
    args = parser.parse_args()
    assert args.test_epoch, 'Test epoch is required.'
    return args

def load_model(model_path):
    print(f'Loading checkpoint from {model_path}')
    model = get_pose_net(cfg, False, JOINT_NUM)
    model = torch.nn.DataParallel(model)
    ckpt = torch.load(model_path, map_location='cpu')
    model.load_state_dict(ckpt['network'])
    model.eval()
    return model

def prepare_input(img_path):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)
    ])
    original_img = cv2.imread(img_path)
    return original_img, transform

def process_image(model, original_img, transform, bbox_list, root_depth_list):
    original_img_height, original_img_width = original_img.shape[:2]
    focal = [1500, 1500]  # x-axis, y-axis
    princpt = [original_img_width/2, original_img_height/2]  # x-axis, y-axis
    
    output_pose_2d_list = []
    output_pose_3d_list = []
    
    for bbox, root_depth in zip(bbox_list, root_depth_list):
        bbox = process_bbox(np.array(bbox), original_img_width, original_img_height)
        img, img2bb_trans = generate_patch_image(original_img, bbox, False, 1.0, 0.0, False)
        img = transform(img).cpu()[None,:,:,:]
        
        with torch.no_grad():
            pose_3d = model(img)[0].cpu().numpy()
        
        pose_3d = post_process_pose(pose_3d, img2bb_trans, root_depth, focal, princpt)
        output_pose_2d_list.append(pose_3d[:,:2].copy())
        output_pose_3d_list.append(pose_3d.copy())
    
    return output_pose_2d_list, output_pose_3d_list

def post_process_pose(pose_3d, img2bb_trans, root_depth, focal, princpt):
    pose_3d[:,0] = pose_3d[:,0] / cfg.output_shape[1] * cfg.input_shape[1]
    pose_3d[:,1] = pose_3d[:,1] / cfg.output_shape[0] * cfg.input_shape[0]
    pose_3d_xy1 = np.concatenate((pose_3d[:,:2], np.ones_like(pose_3d[:,:1])),1)
    img2bb_trans_001 = np.concatenate((img2bb_trans, np.array([0,0,1]).reshape(1,3)))
    pose_3d[:,:2] = np.dot(np.linalg.inv(img2bb_trans_001), pose_3d_xy1.transpose(1,0)).transpose(1,0)[:,:2]
    pose_3d[:,2] = (pose_3d[:,2] / cfg.depth_dim * 2 - 1) * (cfg.bbox_3d_shape[0]/2) + root_depth
    pose_3d = pixel2cam(pose_3d, focal, princpt)
    return pose_3d

def visualize_2d_poses(original_img, output_pose_2d_list, file_name):
    vis_img = original_img.copy()
    for pose_2d in output_pose_2d_list:
        vis_kps = np.zeros((3, JOINT_NUM))
        vis_kps[0,:] = pose_2d[:,0]
        vis_kps[1,:] = pose_2d[:,1]
        vis_kps[2,:] = 1
        vis_img = vis_keypoints(vis_img, vis_kps, SKELETON)
    cv2.imwrite(f'./outputs/{file_name}_2d_output.jpg', vis_img)

def visualize_3d_poses(output_pose_3d_list, file_name):
    vis_kps = np.array(output_pose_3d_list)
    vis_3d_multiple_skeleton(vis_kps, np.ones_like(vis_kps), SKELETON, 'output_pose_3d (x,y,z: camera-centered. mm.)')
    plt.savefig(f'./outputs/{file_name}_3d_output.png')

def calculate_angle(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cos_theta = dot_product / (norm_v1 * norm_v2)
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return np.degrees(angle)

def print_joint_coordinates(output_pose_3d_list):
    for person_idx, person in enumerate(output_pose_3d_list):
        print(f"Person {person_idx + 1}:")
        for index, keypoint in enumerate(person):
            body_part = JOINTS_NAME[index]
            x, y, z = keypoint
            print(f"  {body_part}: x={x:.2f}, y={-y:.2f}, z={z:.2f}")
        print()

def calculate_joint_angles(pose_3d):
    joint_angles = {}
    
    # Define the joint triplets we want to calculate angles for
    joint_triplets = [
        ('Head', 'Thorax', 'R_Shoulder'),
        ('Head', 'Thorax', 'L_Shoulder'),
        ('Thorax', 'R_Shoulder', 'R_Elbow'),
        ('Thorax', 'L_Shoulder', 'L_Elbow'),
        ('R_Shoulder', 'R_Elbow', 'R_Wrist'),
        ('L_Shoulder', 'L_Elbow', 'L_Wrist'),
        ('Thorax', 'Spine', 'Pelvis'),
        ('Spine', 'Pelvis', 'R_Hip'),
        ('Spine', 'Pelvis', 'L_Hip'),
        ('Pelvis', 'R_Hip', 'R_Knee'),
        ('Pelvis', 'L_Hip', 'L_Knee'),
        ('R_Hip', 'R_Knee', 'R_Ankle'),
        ('L_Hip', 'L_Knee', 'L_Ankle'),
    ]
    
    # Create a dictionary mapping joint names to their 3D coordinates
    joint_dict = {JOINTS_NAME[i]: pose_3d[i] for i in range(len(JOINTS_NAME))}
    
    for joint1_name, joint2_name, joint3_name in joint_triplets:
        joint1 = joint_dict[joint1_name]
        joint2 = joint_dict[joint2_name]
        joint3 = joint_dict[joint3_name]
        
        v1 = joint1 - joint2
        v2 = joint3 - joint2
        
        angle = calculate_angle(v1, v2)
        joint_angles[f"{joint1_name}-{joint2_name}-{joint3_name}"] = angle
    
    return joint_angles

def print_joint_angles(output_pose_3d_list):
    for person_idx, pose_3d in enumerate(output_pose_3d_list):
        print(f"Person {person_idx + 1} Joint Angles:")
        joint_angles = calculate_joint_angles(pose_3d)
        for joint_names, angle in joint_angles.items():
            print(f"  {joint_names}: {angle:.2f} degrees")
        print()

def vis_3d_multiple_skeleton_with_angles(kpt_3d, kpt_3d_vis, kps_lines, joint_angles, filename=None):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [np.array((c[2], c[1], c[0])) for c in colors]

    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]

        person_num = kpt_3d.shape[0]
        for n in range(person_num):
            x = np.array([kpt_3d[n,i1,0], kpt_3d[n,i2,0]])
            y = np.array([kpt_3d[n,i1,1], kpt_3d[n,i2,1]])
            z = np.array([kpt_3d[n,i1,2], kpt_3d[n,i2,2]])

            if kpt_3d_vis[n,i1,0] > 0 and kpt_3d_vis[n,i2,0] > 0:
                ax.plot(x, z, -y, c=colors[l], linewidth=2)
            if kpt_3d_vis[n,i1,0] > 0:
                ax.scatter(kpt_3d[n,i1,0], kpt_3d[n,i1,2], -kpt_3d[n,i1,1], c=colors[l], marker='o')
            if kpt_3d_vis[n,i2,0] > 0:
                ax.scatter(kpt_3d[n,i2,0], kpt_3d[n,i2,2], -kpt_3d[n,i2,1], c=colors[l], marker='o')

    # Add joint angle annotations
    for n in range(person_num):
        person_angles = joint_angles[n]
        for joint_name, angle in person_angles.items():
            # Get the middle joint of the angle (usually the vertex of the angle)
            middle_joint = joint_name.split('-')[1]
            joint_index = JOINTS_NAME.index(middle_joint)
            
            # Position the text annotation slightly offset from the joint
            x, y, z = kpt_3d[n, joint_index]
            ax.text(x, z, -y, f'{angle:.1f}°', fontsize=8, color='red')

    if filename is None:
        ax.set_title('3D vis')
    else:
        ax.set_title(filename)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel('Y Label')

    plt.show()
  
def main():
    args = parse_args()
    cfg.set_args(args.gpu_ids)
    model_path = f'./models/snapshot_{int(args.test_epoch)}.pth.tar'
    
    model = load_model(model_path)
    img_path = './inputs/feyd_pose.png'
    
    file_name = osp.splitext(osp.basename(img_path))[0]
    original_img, transform = prepare_input(img_path)
    
    bbox_list = get_bbox_list(img_path)
    root_depth_list = get_root_depth(img_path, bbox_list)
    
    output_pose_2d_list, output_pose_3d_list = process_image(model, original_img, transform, bbox_list, root_depth_list)
    
    # ------------------------ Print and visualize angles ------------------------ 
    # print_joint_coordinates(output_pose_3d_list)
    # print('\n\n')
    # print_joint_angles(output_pose_3d_list)
    
    # visualize_2d_poses(original_img, output_pose_2d_list, file_name)
    # visualize_3d_poses(output_pose_3d_list, file_name)
    
    # Calculate joint angles for all persons
    all_joint_angles = [calculate_joint_angles(pose_3d) for pose_3d in output_pose_3d_list]
    
    # Use the new visualization function
    vis_3d_multiple_skeleton_with_angles(
        np.array(output_pose_3d_list), 
        np.ones_like(output_pose_3d_list), 
        SKELETON, 
        all_joint_angles,
        'output_pose_3d (x,y,z: camera-centered. mm.) with Joint Angles'
    )
    plt.savefig(f'./outputs/{file_name}_3d_output_with_angles.png')

if __name__ == "__main__":
    main()