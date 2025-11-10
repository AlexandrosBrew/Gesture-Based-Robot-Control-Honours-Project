import mediapipe as mp

class baseControl:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
    
    def base_rotation_direction(self, hand_landmarks, handedness, left_thresh=0.4, right_thresh=0.6):
        """Decide base rotation direction based on wrist x-position."""
        wrist_x = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].x
        if wrist_x < left_thresh:
            direction = "LEFT"
            self.rotate_base_left()
        elif wrist_x > right_thresh:
            direction = "RIGHT"
            self.rotate_base_right()
        else:
            direction = "CENTER"
            self.stop_base_rotation()
        return direction
    
    def rotate_base_left(self):
        # print("Rotating base LEFT")
        pass

    def rotate_base_right(self):
        # print("Rotating base RIGHT")
        pass

    def stop_base_rotation(self):
        # print("Stopping base rotation")
        pass