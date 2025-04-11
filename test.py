import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

def extract_clothing(source_image_path, target_image_path):
    # Load images
    source_image = cv2.imread(source_image_path)
    target_image = cv2.imread(target_image_path)

    # Convert to RGB
    source_rgb = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
    target_rgb = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)

    # Process the source image to find pose landmarks
    results = pose.process(source_rgb)

    if results.pose_landmarks:
        # Extract clothing area based on landmarks (e.g., shoulders, hips)
        # Define bounding box coordinates (example values)
        h, w, _ = source_image.shape
        x_min = int(w * 0.3)
        x_max = int(w * 0.7)
        y_min = int(h * 0.4)
        y_max = int(h * 0.8)

        # Create a mask for the clothing
        clothing_mask = np.zeros_like(source_image)
        clothing_mask[y_min:y_max, x_min:x_max] = source_image[y_min:y_max, x_min:x_max]

        # Overlay the clothing onto the target image
        target_image[y_min:y_max, x_min:x_max] = clothing_mask[y_min:y_max, x_min:x_max]

    # Save or display the result
    cv2.imshow('Result', target_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
extract_clothing('source_image.jpg', 'target_image.jpg')