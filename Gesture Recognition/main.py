import cv2
import mediapipe as mp
import tracking
import vidCapture

def main():
    cap = vidCapture.Capture()
    tracker = tracking.HandTracker(detection_confidence=0.8, tracking_confidence=0.5)

    while True:
        success, frame = cap.read_frame()
        if success:
            results = tracker.process_frame(frame)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    tracker.draw_landmarks(frame, hand_landmarks)
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) == ord('q'):
                print("Quitting...")
                break
    cap.release()
    
if __name__ == "__main__":
    main()
