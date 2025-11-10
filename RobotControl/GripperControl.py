import math
import cv2

class GripperControl:
    def __init__(self):
        self.previous_angle = 0
        
    def FingerDistance(self, handLandmarks, handedness):
        """Calculate Euclidean distance between two landmarks."""
        if handedness.classification[0].label == "Left":
            # Swap landmarks for left hand to maintain consistency
            landmark1 = handLandmarks.landmark[8]  # Index finger tip
            landmark2 = handLandmarks.landmark[4]  # Thumb tip
            return math.sqrt((landmark1.x - landmark2.x) ** 2 +
                            (landmark1.y - landmark2.y) ** 2 +
                            (landmark1.z - landmark2.z) ** 2)
        return 0

    def is_finger_pinched(self, distance, threshold=0.065):
        """Determine if fingers are pinched based on distance threshold."""
        return distance < threshold

    def normalise_distance(self, distance, min_dist=0.02, max_dist=0.2, alpha=0.5):
        """Normalize distance to range [0, 1] and scale to servo angle [0, 180]. Applies smoothing."""
        openness = max(min_dist, min(distance, max_dist))
        normalized = (openness - min_dist) / (max_dist - min_dist)
        target_angle = int(normalized * 180)
        if self.is_finger_pinched(distance):
            target_angle = 0  # Fully closed
        smoothed_angle = int(alpha * target_angle + (1 - alpha) * self.previous_angle)
        self.previous_angle = smoothed_angle
        return smoothed_angle

    def draw_gripper_status(self, frame, angle, position=(200, 450)):
        """Overlay gripper status on the video frame."""
        status_text = f"Gripper Angle: {angle} degrees"
        cv2.putText(frame, status_text, position,
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Example usage:
# if results.multi_hand_landmarks and results.multi_handedness:
#             for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
#                 distance = gripper_control.FingerDistance(hand_landmarks, handedness)
#                 angle = gripper_control.normalise_distance(distance)
#                 print(f"Hand: {handedness.classification[0].label}, Finger Distance: {distance:.4f}, Servo Angle: {angle}")
