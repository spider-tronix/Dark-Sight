from cv2 import cv2


class PiCamera:
    def __init__(self):

        print("Connecting to PI cam...")
        self.cap = cv2.VideoCapture("udp://192.168.0.107:1234", cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        while not self.cap.isOpened():
            cv2.waitKey(10)

        print("Pi cam connected!")

    def __call__(self):

        _, frame = self.cap.read()

        # cv2.imshow('image here', frame)
        return frame

    def release(self):
        self.cap.release()


def main():
    picam = PiCamera()
    while True:

        cv2.imshow("img", picam())

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    picam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
