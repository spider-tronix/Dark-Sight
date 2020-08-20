from cv2 import cv2
cap = cv2.VideoCapture('udp://192.168.0.104:1234',cv2.CAP_FFMPEG)
while not cap.isOpened():
    print('VideoCapture not opened')
    cv2.waitKey(10)

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