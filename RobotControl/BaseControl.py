import mediapipe as mp

class baseControl:
    def __init__(self, Comm):
        self.mp_hands = mp.solutions.hands
        self.Comm = Comm
    
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
        #Move up by 10
        self.Comm.send_servo_command(3, 10)

    def rotate_base_right(self):
        self.Comm.send_servo_command(3, -10)

    def stop_base_rotation(self):
        self.Comm.send_servo_command(3, 0)