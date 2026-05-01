import mediapipe as mp
from RobotControl.utils import angle_to_microseconds
from RobotControl.PID import PIDController

class WristControl:
    def __init__(self, Comm):
        self.mp_hands = mp.solutions.hands
        self.Comm = Comm
        
        # Initialize PID: target is 0.5 (center). 
        # You will need to tune these Kp, Ki, Kd values!
        self.pid = PIDController(kp=30.0, ki=0.0, kd=5.0, setpoint=0.5)
        # Keep track of the absolute angle (assuming 0-180 degree servo)
        self.current_angle = 90.0 

    def update_wrist_position(self, hand_landmarks):
        """Update wrist rotation dynamically using PID."""
        wrist_x = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].x

        # 1. Get the required angle adjustment from the PID controller
        adjustment = self.pid.compute(wrist_x)

        # 2. Apply adjustment to current angle
        # (Subtracting because if wrist_x > 0.5, error is negative, but we might want positive rotation)
        self.current_angle -= adjustment 

        # 3. Clamp the angle to physical servo limits (0 to 180 degrees)
        self.current_angle = max(0.0, min(180.0, self.current_angle))

        return angle_to_microseconds(self.current_angle)