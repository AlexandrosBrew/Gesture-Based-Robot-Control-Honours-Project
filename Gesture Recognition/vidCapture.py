import cv2

class Capture:
    def __init__(self, camera_index=0, width=640, height=480):
        self.capture = cv2.VideoCapture(camera_index)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read_frame(self):
        success, frame = self.capture.read()
        return success, frame

    def release(self):
        self.capture.release()
        cv2.destroyAllWindows()

# Example usage:
def main():
    cap = Capture()
    while True:
        success, frame = cap.read_frame()
        if success:
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) == ord('q'):
                print("Quitting...")
                break
    cap.release()

if __name__ == "__main__":
    main()