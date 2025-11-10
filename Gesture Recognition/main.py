import cv2
import mediapipe as mp
import tracking
import vidCapture
from OneEuroFilter import OneEuroFilter
from mediapipe.framework.formats import landmark_pb2

def main():
    cap = vidCapture.Capture()
    tracker = tracking.HandTracker(detection_confidence=0.8, tracking_confidence=0.5)

    print("Starting hand tracking... Press 'q' to quit.")

    while True:
        success, frame = cap.read_frame()
        frame = cv2.flip(frame, 1)  # Mirror for selfie-view

        if not success:
            break

        results = tracker.process_frame(frame)

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                tracker.draw_landmarks(frame, hand_landmarks, handedness)

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting...")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
