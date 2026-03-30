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
            cv2.line(frame, (int(w*0.65), 0), (int(w*0.65), h), (0, 0, 255), 2)
            cv2.line(frame, (int(w*0.8), 0), (int(w*0.8), h), (0, 0, 255), 2)
        elif handedness.classification[0].label == "Right":
            # Visual Up/center/down zones
            cv2.line(frame, (0, int(h*0.3)), (w*2, int(h*0.3)), (0, 0, 255), 2)
            cv2.line(frame, (0, int(h*0.45)), (w*2, int(h*0.45)), (0, 0, 255), 2)

    def process_left_hand(self, frame, hand_landmarks, handedness):
        distance = self.gripper_control.FingerDistance(hand_landmarks, handedness)
        angle = self.gripper_control.normalise_distance(distance)
        self.gripper_control.draw_gripper_status(frame, angle)
        
        self.comm.send_servo_command(0, int(angle))  # Send gripper angle to Arduino
        
        direction = self.base_control.update_base_position(hand_landmarks)
        cv2.putText(frame, f"Base: {direction}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    def process_right_hand(self, frame, hand_landmarks, handedness):
        distance = self.elbow_control.elbow_angle(hand_landmarks, handedness)
        elbow_angle = self.elbow_control.normalise_distance(distance)
        self.elbow_control.draw_elbow_status(frame, elbow_angle)
        self.comm.send_servo_command(2, int(elbow_angle))  # Send elbow angle to Arduino
        base_angle = self.wrist_control.wrist_position(hand_landmarks, handedness)
        
        cv2.putText(frame, f"Wrist: {base_angle}", (10, 80),
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
            home_position = [0, 90, 90, 90]
            ik = self.kinematics_solver.inverse(home_position[0], home_position[1], home_position[2], home_position[3])
            if ik:
                limits = self.kinematics_solver.LIMITS
        
                # Map each joint using its specific limits
                pwm_base     = map_angle_to_pwm(ik.theta1, *limits["theta1"])
                pwm_shoulder = map_angle_to_pwm(ik.theta2, *limits["theta2"])
                pwm_elbow    = map_angle_to_pwm(ik.theta3, *limits["theta3"])
                pwm_wrist    = map_angle_to_pwm(ik.theta4, *limits["theta4"])

                # Send the mapped PWM values
                self.comm.send_servo_command(0, pwm_base)
                self.comm.send_servo_command(1, pwm_shoulder)
                self.comm.send_servo_command(2, pwm_elbow)
                self.comm.send_servo_command(3, pwm_wrist)
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

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    controller = HandTrackingController()
    controller.run()
