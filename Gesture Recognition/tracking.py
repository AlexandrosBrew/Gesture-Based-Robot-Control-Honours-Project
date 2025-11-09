import mediapipe as mp
import cv2
import vidCapture

class HandTracker:
    def __init__(self, detection_confidence=0.7, tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=detection_confidence, min_tracking_confidence=tracking_confidence)
        self.mp_drawing = mp.solutions.drawing_utils

    def process_frame(self, frame):
        RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(RGB_frame)
        return results

    def draw_landmarks(self, frame, hand_landmarks):
        self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        


# Example usage:
def main():
    cap = vidCapture.Capture()
    tracker = HandTracker(detection_confidence=0.8, tracking_confidence=0.5)

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