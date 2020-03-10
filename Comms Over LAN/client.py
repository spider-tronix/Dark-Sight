import cv2
import numpy as np
import zmq

context = zmq.Context()
footage_socket = context.socket(zmq.SUB)
footage_socket.bind('udp://*:2000')
footage_socket.setsockopt_string(zmq.SUBSCRIBE, np.unicode(''))

while True:
    try:
        frame = footage_socket.recv_string()
        # img = base64.b64decode(frame)
        # npimg = np.fromstring(img, dtype=np.uint8)
        # source = cv2.imdecode(npimg, 1)
        # cv2.imshow("Stream", source)
        # cv2.waitKey(1)
        print(frame)

    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        break
