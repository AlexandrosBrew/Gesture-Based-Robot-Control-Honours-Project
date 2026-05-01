import cv2
import time
import sys
sys.path.append('../')
import GestureRecognition.gestureRecognition as gestRecognition
import GestureRecognition.tracking as tracking
import GestureRecognition.vidCapture as vidCapture
from RobotControl.GripperControl import GripperControl
from RobotControl.BaseControl import baseControl
from RobotControl.ElbowControl import ElbowControl
from RobotControl.WristControl import WristControl
from RobotControl.kinematicsSolver import RobotKinematics, IKResult
from RobotControl.utils import angle_to_microseconds, map_angle_to_pwm
from Comms import Comm
class HandTrackingController:
    def __init__(self):
        self.comm = Comm()
        self.cap = vidCapture.Capture(width=640, height=480)
        self.tracker = tracking.HandTracker(detection_confidence=0.8, tracking_confidence=0.5)
        self.gripper_control = GripperControl()
        self.base_control = baseControl(self.comm)
        self.elbow_control = ElbowControl(self.comm)
        self.wrist_control = WristControl(self.comm)
        self.gestRecognition = gestRecognition.GestureRecognition(self.tracker)
        self.kinematics_solver = RobotKinematics()
        self.control_mode = 'View'
        print("Starting hand tracking... Press 'q' to quit.")
        self.prev_time = time.time()
        self.base_angle = 1500;
        self.wrist_angle = 1500;
        self.elbow_angle = 1000;
        self.gripper_angle = 500;

    def process_key(self, key):
        if key == ord('1'):
            self.control_mode = 'View'
            print("Control mode set to View.")
        elif key == ord('2'):
            self.control_mode = 'Control'
            print("Control mode set to Control.")
        elif key == ord('3'):
            self.control_mode = 'Gesture'
            print("Control mode set to Gesture.")
        elif key == ord('q'):
            print("Quitting...")
            return False
        return True

    def draw_mode(self, frame):
        mode_text = f"{self.control_mode.upper()}"
        color = (0, 255, 0) if self.control_mode == 'Control' or self.control_mode == 'Gesture' else (0, 0, 255)
        cv2.putText(frame, mode_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    def draw_visual_markers(self, frame, handedness):
        w, h, _ = frame.shape
        if handedness.classification[0].label == "Left":
            # Visual Left/center/right zones
            cv2.line(frame, (int(w*0.5), 0), (int(w*0.5), h), (0, 0, 255), 2)
            cv2.line(frame, (int(w*0.65), 0), (int(w*0.65), h), (0, 255, 250), 2)
            cv2.line(frame, (int(w*0.8), 0), (int(w*0.8), h), (255, 0, 0), 2)
        elif handedness.classification[0].label == "Right":
            # Visual Up/center/down zones
            cv2.line(frame, (int(w*0.5), 0), (int(w*0.5), h), (0, 0, 255), 2)
            cv2.line(frame, (int(w*0.65), 0), (int(w*0.65), h), (0, 255, 0), 2)
            cv2.line(frame, (int(w*0.8), 0), (int(w*0.8), h), (255, 0, 0), 2)

    def process_left_hand(self, frame, hand_landmarks, handedness):
        distance = self.gripper_control.FingerDistance(hand_landmarks, handedness)
        self.gripper_angle = self.gripper_control.normalise_distance(distance)
        self.gripper_control.draw_gripper_status(frame, self.gripper_angle)
        
        self.comm.send_servo_command(0, int(self.gripper_angle))  # Send gripper angle to Arduino
        self.base_angle = self.base_control.update_base_position(hand_landmarks)
        cv2.putText(frame, f"Base: {self.base_angle:.2f}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    def process_right_hand(self, frame, hand_landmarks, handedness):
        distance = self.elbow_control.elbow_angle(hand_landmarks, handedness)
        self.elbow_angle = self.elbow_control.normalise_distance(distance)
        self.elbow_control.draw_elbow_status(frame, self.elbow_angle)
        self.wrist_angle = self.wrist_control.update_wrist_position(hand_landmarks)
        cv2.putText(frame, f"Wrist: {self.wrist_angle:.2f}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    def process_hand(self, frame, hand_landmarks, handedness):
        if handedness.classification[0].label == "Left":
            self.process_left_hand(frame, hand_landmarks, handedness)

        elif handedness.classification[0].label == "Right":
            self.process_right_hand(frame, hand_landmarks, handedness)

        self.draw_visual_markers(frame, handedness)
        self.tracker.draw_landmarks(frame, hand_landmarks, handedness)

    def process_gesture(self, frame, hand_landmarks, handedness):
        gesture_name = self.gestRecognition.recognize_gesture(hand_landmarks, handedness)
        if gesture_name == "Thumbs Up":
            cv2.putText(frame, f"Gesture: {gesture_name}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            #Move robot to home position
            home_position = [150, 0, 100, True]  # Example home position (x, y, z, elbow_up)
            ik = self.kinematics_solver.inverse(home_position[0], home_position[1], home_position[2], home_position[3])
            # Map each joint using its specific limits
            if ik is not None:
                pwm_base = angle_to_microseconds(ik.theta1)  
                pwm_wrist = angle_to_microseconds(ik.theta2)  
                pwm_elbow = angle_to_microseconds(ik.theta3)  
                self.base_angle = pwm_base
                self.wrist_angle = pwm_wrist
                self.elbow_angle = pwm_elbow
                self.gripper_angle = 500  # Fully closed for home position

    def sendcommands(self):
        self.comm.send_servo_command(0, int(self.gripper_angle))  # Gripper
        self.comm.send_servo_command(1, int(self.wrist_angle))    # Wrist
        self.comm.send_servo_command(2, int(self.elbow_angle))    # Elbow
        self.comm.send_servo_command(3, int(self.base_angle))     # Base

    def run(self):
        while True:
            success, frame = self.cap.read_frame()
            if not success:
                break

            frame = cv2.flip(frame, 1)  # Mirror for selfie-view
            key = cv2.waitKey(1) & 0xFF

            if not self.process_key(key):
                break

            results = self.tracker.process_frame(frame)
            if self.control_mode == 'Control':
                if results.multi_hand_landmarks and results.multi_handedness:
                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                        self.process_hand(frame, hand_landmarks, handedness)
            if self.control_mode == 'Gesture':
                if results.multi_hand_landmarks and results.multi_handedness:
                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                        self.tracker.draw_landmarks(frame, hand_landmarks, handedness)
                        self.process_gesture(frame, hand_landmarks, handedness)

            self.draw_mode(frame)
            # Calculate and draw FPS (top-right)
            # curr_time = time.time()
            # fps = 1.0 / (curr_time - self.prev_time) if (curr_time - self.prev_time) > 0 else 0.0
            # self.prev_time = curr_time
            # h, w = frame.shape[:2]
            # fps_text = f"FPS: {fps:.1f}"
            # (text_w, text_h), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            # x = w - text_w - 10
            # y = 30
            # cv2.putText(frame, fps_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Frame", frame)
            # send commands to Arduino at the end of each loop iteration
            self.sendcommands()

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    controller = HandTrackingController()
    controller.run()
