from cv2 import cv2


def initialize():

    print('Connecting to PI cam...')
    cap = cv2.VideoCapture('udp://192.168.0.104:1234',cv2.CAP_FFMPEG)
    while not cap.isOpened():
        
        cv2.waitKey(10)
    print('Pi cam connected!')
    return cap


def pi_img(cap):

    ret, frame = cap.read()

    # cv2.imshow('image here', frame)

    return frame
    

def main():
    cap = initialize()
    while True:
        pi_img(cap)

        if cv2.waitKey(1)&0XFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__": 
    main()
