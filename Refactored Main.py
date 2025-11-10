import cv2
import sys
sys.path.append('../')
import GestureRecognition.tracking as tracking
import GestureRecognition.vidCapture as vidCapture
from RobotControl.GripperControl import GripperControl
from RobotControl.BaseControl import baseControl
from RobotControl.ElbowControl import ElbowControl
from RobotControl.WristControl import WristControl

class HandTrackingController:
    def __init__(self):
        self.cap = vidCapture.Capture()
        self.tracker = tracking.HandTracker(detection_confidence=0.8, tracking_confidence=0.5)
        self.gripper_control = GripperControl()
        self.base_control = baseControl()
        self.elbow_control = ElbowControl()
        self.wrist_control = WristControl()
        self.control_mode = 'View'
        print("Starting hand tracking... Press 'q' to quit.")

    def process_key(self, key):
        if key == ord('1'):
            self.control_mode = 'View'
            print("Control mode set to View.")
        elif key == ord('2'):
            self.control_mode = 'Control'
            print("Control mode set to Control.")
        elif key == ord('q'):
            print("Quitting...")
            return False
        return True

    def draw_mode(self, frame):
        mode_text = f"{self.control_mode.upper()}"
        color = (0, 255, 0) if self.control_mode == 'Control' else (0, 0, 255)
        cv2.putText(frame, mode_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    def draw_visual_markers(self, frame, handedness):
        w, h, _ = frame.shape
        if handedness.classification[0].label == "Left":
            # Visual Left/center/right zones
            cv2.line(frame, (int(w*0.5), 0), (int(w*0.5), h), (0, 0, 255), 2)
            cv2.line(frame, (int(w*0.8), 0), (int(w*0.8), h), (0, 0, 255), 2)
        elif handedness.classification[0].label == "Right":
            # Visual Up/center/down zones
            cv2.line(frame, (0, int(h*0.3)), (w*2, int(h*0.3)), (0, 0, 255), 2)
            cv2.line(frame, (0, int(h*0.45)), (w*2, int(h*0.45)), (0, 0, 255), 2)

    def process_hand(self, frame, hand_landmarks, handedness):
        if self.control_mode != 'Control':
            return

        if handedness.classification[0].label == "Left":
            distance = self.gripper_control.FingerDistance(hand_landmarks, handedness)
            angle = self.gripper_control.normalise_distance(distance)
            self.gripper_control.draw_gripper_status(frame, angle)

            direction = self.base_control.base_rotation_direction(hand_landmarks, handedness)
            cv2.putText(frame, f"Base: {direction}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        elif handedness.classification[0].label == "Right":
            distance = self.elbow_control.elbow_angle(hand_landmarks, handedness)
            angle = self.elbow_control.normalise_distance(distance)
            self.elbow_control.draw_elbow_status(frame, angle)

            direction = self.wrist_control.wrist_position(hand_landmarks, handedness)
            cv2.putText(frame, f"Wrist: {direction}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        self.draw_visual_markers(frame, handedness)
        self.tracker.draw_landmarks(frame, hand_landmarks, handedness)

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

            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    self.process_hand(frame, hand_landmarks, handedness)

            self.draw_mode(frame)
            cv2.imshow("Frame", frame)

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    controller = HandTrackingController()
    controller.run()
