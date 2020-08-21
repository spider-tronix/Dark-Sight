from cv2 import cv2

def main():
    cap = cv2.VideoCapture('udp://192.168.0.104:1234',cv2.CAP_FFMPEG)
    print('Opening PI cam...')
    while not cap.isOpened():
        
        cv2.waitKey(10)
    print('Pi cam connected!')
    while True:
        ret, frame = cap.read()

        if not ret:
            print('frame empty')
            break

        cv2.imshow('image', frame)

        if cv2.waitKey(1)&0XFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__": 
    main()
