import cv2

class Capture:
    def __init__(self, camera_index=0, width=1024, height=720):
        self.capture = cv2.VideoCapture(camera_index)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read_frame(self):
        success, frame = self.capture.read()
        return success, frame

    def release(self):
        self.capture.release()
        cv2.destroyAllWindows()