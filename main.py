from src.pose_estimation.pose_estimator import PoseEstimator
from src.angle_calculation.angle_calculator import AngleCalculator
from src.risk_assessment.rula import RULAAssessment
import yaml

def main():
    # Load configuration
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize components
    pose_estimator = PoseEstimator(config['pose_estimation'])
    angle_calculator = AngleCalculator()
    risk_assessor = RULAAssessment()
    
    # Process video/image
    input_path = config['input_path']
    results = pose_estimator.process(input_path)
    
    # Calculate angles and assess risk
    for frame_poses in results:
        # Calculate joint angles
        angles = angle_calculator.calculate_angles(frame_poses)
        
        # Perform risk assessment
        risk_score = risk_assessor.assess(angles)
        
        # Visualize and store results
        # ...

if __name__ == "__main__":
    main()