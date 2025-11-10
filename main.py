import cv2
import sys
sys.path.append('../')
import GestureRecognition.tracking as tracking
import GestureRecognition.vidCapture as vidCapture
from RobotControl.GripperControl import GripperControl
from RobotControl.BaseControl import baseControl
from RobotControl.ElbowControl import ElbowControl
from RobotControl.WristControl import WristControl

def main():
    cap = vidCapture.Capture()
    tracker = tracking.HandTracker(detection_confidence=0.8, tracking_confidence=0.5)
    gripper_control = GripperControl()
    base_control = baseControl()
    elbow_control = ElbowControl()
    wrist_control = WristControl()
    print("Starting hand tracking... Press 'q' to quit.")
    control_mode = 'View'

    while True:
        success, frame = cap.read_frame()
        frame = cv2.flip(frame, 1)  # Mirror for selfie-view
        key = cv2.waitKey(1) & 0xFF
        
        if not success:
            break
        
        if key == ord('1'):
            control_mode = 'View'
            print("Control mode set to View.")
        elif key == ord('2'):
            control_mode = 'Control'
            print("Control mode set to Control.")
        elif key == ord('q'):
            print("Quitting...")
            break

        results = tracker.process_frame(frame)

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                if control_mode == 'Control':

                    if handedness.classification[0].label == "Left":
                        distance = gripper_control.FingerDistance(hand_landmarks, handedness)
                        angle = gripper_control.normalise_distance(distance)
                        gripper_control.draw_gripper_status(frame, angle)
                        direction = base_control.base_rotation_direction(hand_landmarks, handedness)
                        cv2.putText(frame, f"Base: {direction}", (10, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        # Add visual markers for left/center/right zones
                        w, h, _ = frame.shape
                        cv2.line(frame, (int(w*0.5), 0), (int(w*0.5), h), (0, 0, 255), 2)
                        cv2.line(frame, (int(w*0.8), 0), (int(w*0.8), h), (0, 0, 255), 2)
                    elif handedness.classification[0].label == "Right":
                        distance = elbow_control.elbow_angle(hand_landmarks, handedness)
                        angle = elbow_control.normalise_distance(distance)
                        elbow_control.draw_elbow_status(frame, angle)
                        direction = wrist_control.wrist_position(hand_landmarks, handedness)
                        cv2.putText(frame, f"Wrist: {direction}", (10, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        #ADd visual markers for up/center/down zones
                        w, h, _ = frame.shape
                        cv2.line(frame, (0, int(h*0.3)), (w*2, int(h*0.3)), (0, 0, 255), 2)
                        w, h, _ = frame.shape
                        cv2.line(frame, (0, int(h*0.45)), (w*2, int(h*0.45)), (0, 0, 255), 2)
                tracker.draw_landmarks(frame, hand_landmarks, handedness)
        
        
        mode_text = f"{control_mode.upper()}"
        color = (0, 255, 0) if control_mode else (0, 0, 255)
        cv2.putText(frame, mode_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("Frame", frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
