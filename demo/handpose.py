import mediapipe as mp
import cv2
import os

def process_hand_keypoints(image_path, file_name):
    # Initialize MediaPipe hands module
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    # Read the input image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to read image at {image_path}")

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and detect hands
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
        results = hands.process(image_rgb)

    # Initialize list to store keypoints
    keypoints = []

    # Check if hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract keypoints
            for landmark in hand_landmarks.landmark:
                keypoints.append((landmark.x, landmark.y, landmark.z))

            # Draw landmarks on the image
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
            )

    # Create output directory if it doesn't exist
    os.makedirs("./outputs", exist_ok=True)

    # Save the annotated image
    output_path = os.path.join("./outputs", f"{file_name}_handpose.jpg")
    cv2.imwrite(output_path, image)

    return keypoints

def process_body_keypoints(image_path):
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    # Read the input image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to read image at {image_path}")

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and detect pose
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        results = pose.process(image_rgb)

    # Initialize dictionary to store keypoints
    keypoints = {}

    # Check if pose is detected
    if results.pose_landmarks:
        # Draw landmarks on the image
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Extract keypoints
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            keypoints[mp_pose.PoseLandmark(idx).name] = (landmark.x, landmark.y, landmark.z)

    # Create output directory if it doesn't exist
    os.makedirs("./outputs", exist_ok=True)

    # Save the annotated image
    file_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join("./outputs", f"{file_name}_bodypose.jpg")
    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    return keypoints

# Example usage
if __name__ == "__main__":
    input_image_path = "./inputs/input_old_man.jpeg"
    
    # Process hand keypoints
    hand_keypoints = process_hand_keypoints(input_image_path, 'old_man')
    print(f"Hand keypoints: {hand_keypoints}")

    # Process body keypoints
    body_keypoints = process_body_keypoints(input_image_path)
    print(f"Body keypoints: {body_keypoints}")
    
