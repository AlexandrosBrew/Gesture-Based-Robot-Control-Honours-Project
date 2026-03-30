class GestureRecognition:
    def __init__(self, tracker):
        self.tracker = tracker

    def recognize_gesture(self, hand_landmarks, handedness):
        """Simple rule-based gesture recognition for demonstration purposes.
        
        -----------------------------
        Returns:
        - "Thumbs Up" if the gesture is recognized.
        - "Unknown" if the gesture does not match any known patterns.
        -----------------------------
        """
        #check if thumbs up
        if handedness.classification[0].label == "Right":
            thumb_tip = hand_landmarks.landmark[4]
            index_mcp = hand_landmarks.landmark[5]
            if (thumb_tip.y < index_mcp.y and  # Simple check for thumbs up
                hand_landmarks.landmark[8].y > index_mcp.y and  # Index finger down
                hand_landmarks.landmark[12].y > index_mcp.y and  # Middle finger down
                hand_landmarks.landmark[16].y > index_mcp.y and  # Ring finger down
                hand_landmarks.landmark[20].y > index_mcp.y):  # Pinky down
                return "Thumbs Up" 
        return "Unknown"  # Default case for unrecognized gestures
    
    