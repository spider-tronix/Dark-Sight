from cv2 import cv2
import threading

class CameraBufferCleanerThread(threading.Thread):
    def __init__(self, camera, name='camera-buffer-cleaner-thread'):
        self.camera = camera
        self.last_frame = None
        super(CameraBufferCleanerThread, self).__init__(name=name)
        self.start()

    def run(self):
        while True:
            ret, self.last_frame = self.camera.read()

class PiCamera:
    def __init__(self):

        print("Connecting to PI cam...")
        self.cap = cv2.VideoCapture("udp://192.168.0.107:1234", cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cam_cleaner = CameraBufferCleanerThread(self.cap)
        while not self.cap.isOpened():
            cv2.waitKey(10)

        print("Pi cam connected!")

    def __call__(self):

        frame = self.cam_cleaner.last_frame

        # cv2.imshow('image here', frame)
        return frame

    def release(self):
        self.cap.release()


def main():
    picam = PiCamera()
    while True:
        try:
            cv2.imshow("img", picam())
        except:
            pass

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    picam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
