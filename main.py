from pose_estimator import PoseEstimator
from angle_calculator import AngleCalculator
from rula import calculate_rula_score

def main():
    # Input image path
    input_image = "./inputs/rula_test_girl.jpg"
    
    # Initialize components
    pose_estimator = PoseEstimator()
    angle_calculator = AngleCalculator()
    
    # Get pose keypoints
    keypoints_dict = pose_estimator.estimate_pose(input_image)
    
    # Calculate angles from keypoints
    angles = angle_calculator.calculate_angles(keypoints_dict)
    
    # Calculate RULA scores
    rula_scores = calculate_rula_score(angles)
    
    # Print results
    print("\n=== RULA Assessment Results ===")
    print(f"Final RULA Score: {rula_scores['final_score']}")
    print("\nDetailed Angles:")
    for angle_name, angle_value in angles.items():
        print(f"{angle_name}: {angle_value:.1f}°")
    
    print("\nDetailed Scores:")
    for score_name, score_value in rula_scores.items():
        if score_name != 'final_score':  # Skip final score as it's already printed
            print(f"{score_name}: {score_value}")

if __name__ == "__main__":
    main()
