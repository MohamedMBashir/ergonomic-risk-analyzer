from mmpose.apis import MMPoseInferencer
import numpy as np

class PoseEstimator:
    def __init__(self, config):
        self.inferencer = MMPoseInferencer(
            pose3d='human3d',
            device=config.get('device', 'cuda:0')
        )
    
    def process(self, input_path):
        """Process input and return 3D poses"""
        result_generator = self.inferencer(input_path)
        poses = []
        
        try:
            while True:
                result = next(result_generator)
                frame_poses = self._extract_poses(result)
                poses.append(frame_poses)
        except StopIteration:
            pass
            
        return poses
    
    def _extract_poses(self, result):
        """Extract relevant pose data from MMPose output"""
        return result['predictions'][0]