import mediapipe as mp

class WristControl:
    def __init__(self, Comm):
        self.mp_hands = mp.solutions.hands
        self.Comm = Comm

    def wrist_position(self, hand_landmarks, handedness, up_thresh=0.4, down_thresh=0.6):
        """Get wrist position coordinates."""
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        if wrist.y < up_thresh:
            position = "UP"
            self.move_wrist_up()
        elif wrist.y > down_thresh:
            position = "DOWN"
            self.move_wrist_down()
        else:
            position = "CENTER"
            self.stop_wrist_movement()
        return position

    def move_wrist_up(self):
        #Move up by 10
        self.Comm.send_servo_command(1, 10)

    def move_wrist_down(self):
        self.Comm.send_servo_command(1, -10)

    def stop_wrist_movement(self):
        # print("Stopping wrist movement")
        self.Comm.send_servo_command(1, 0)