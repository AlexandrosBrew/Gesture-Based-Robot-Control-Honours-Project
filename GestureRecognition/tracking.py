import mediapipe as mp
import cv2
from OneEuroFilter import OneEuroFilter
from mediapipe.framework.formats import landmark_pb2

class HandTracker:
    def __init__(self, detection_confidence=0.7, tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=detection_confidence, min_tracking_confidence=tracking_confidence, max_num_hands=2, model_complexity=1, static_image_mode=False)
        self.mp_drawing = mp.solutions.drawing_utils
        # Initialize separate OneEuro filters for both hands
        # 21 landmarks × 3 coordinates (x, y, z)
        self.filters_left = [[OneEuroFilter(freq=10, mincutoff=1.0, beta=0.0, dcutoff=1.0) for _ in range(3)] for _ in range(21)]
        self.filters_right = [[OneEuroFilter(freq=10, mincutoff=1.0, beta=0.0, dcutoff=1.0) for _ in range(3)] for _ in range(21)]

    def process_frame(self, frame):
        RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(RGB_frame)
        return results

    def draw_landmarks(self, frame, hand_landmarks, handedness):
        label = handedness.classification[0].label  # "Left" or "Right"

        # Apply different filters for each hand
        if label == "Right":
            smoothed_landmarks = self.smooth_one_euro(hand_landmarks.landmark, self.filters_right)
        else:
            smoothed_landmarks = self.smooth_one_euro(hand_landmarks.landmark, self.filters_left)

        self.mp_drawing.draw_landmarks(frame, smoothed_landmarks, self.mp_hands.HAND_CONNECTIONS)

        # Draw label
        h, w, _ = frame.shape
        wrist = smoothed_landmarks.landmark[0]
        cx, cy = int(wrist.x * w), int(wrist.y * h)
        cv2.putText(frame, label, (cx - 20, cy - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    def smooth_one_euro(self, landmarks, filters):
        """Apply OneEuro smoothing to a Mediapipe landmark list using a given filter set."""
        smoothed = []
        for i, lm in enumerate(landmarks):
            x = filters[i][0](lm.x)
            y = filters[i][1](lm.y)
            z = filters[i][2](lm.z)
            smoothed.append(landmark_pb2.NormalizedLandmark(x=x, y=y, z=z))
        return landmark_pb2.NormalizedLandmarkList(landmark=smoothed)