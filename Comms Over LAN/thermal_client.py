import socket
import time

import cv2
import numpy as np
import paramiko


class Netcat:
    """ Python 'netcat like' module """

    def __init__(self, ip, port):
        self.buff = ""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((ip, port))
        self.conn = None

    def listen(self):
        self.socket.listen(2)
        self.conn, addr = self.socket.accept()

    def read(self, length=1024):
        """ Read 1024 bytes off the socket """
        return self.conn.recv(length)

    def read_until(self, data):
        """ Read data into the buffer until we have data """
        while data not in self.buff:
            self.buff += self.conn.recv(1024).decode('ascii')

        pos = self.buff.find(data)
        rval = self.buff[:pos + len(data)]
        self.buff = self.buff[pos + len(data):]

        return rval

    def write(self, data):
        self.socket.send(data)

    def close(self):
        self.socket.close()


class ThermalCamera:
    def __init__(self):
        self.port = '22'
        self.uname = 'pi'
        self.passd = 'sharan'
        self.ip = '192.168.43.38'
        self.pi_ssh = paramiko.SSHClient()
        self.connect_ssh()

    def connect_ssh(self):
        self.pi_ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.pi_ssh.connect(hostname=self.ip, username=self.uname, password=self.passd)

    def trigger_camera(self):
        _ = self.pi_ssh.exec_command("./test | nc 192.168.43.156 2000")
        # _ = self.pi_ssh.exec_command("timeout 3 fbuf")


def stdout2arr(string):
    tem_row = list(string.strip().split('\n'))
    temp = []
    for col in tem_row:
        temp.append(list(map(float, col.strip().split(' '))))
    temp = np.array(temp, dtype=np.float32)

    temp = np.flipud(temp)
    temp = np.fliplr(temp)
    return temp


def main():
    nc = Netcat('192.168.43.156', 2000)
    cam = ThermalCamera()
    cam.trigger_camera()
    nc.listen()
    _ = nc.read_until('End')
    thermal = 'Thermal Feed'
    cv2.namedWindow(thermal, cv2.WINDOW_NORMAL)
    while True:
        tick = time.time()

        data = nc.read_until('End')
        data = data[data.find('Subpage:') + 11:-4]
        proc = stdout2arr(data)

        tock = time.time()
        print(-(tick - tock))

        # proc = np.interp(proc, [np.min(proc) - 1, np.max(proc) + 1], [0, 255])
        proc = np.interp(proc, [20 - 1, 36 + 1], [0, 255])
        vis = cv2.cvtColor(proc.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        vis = cv2.resize(vis, (960, 720))

        cv2.imshow(thermal, vis)
        ch = cv2.waitKey(1)
        if ch == ord('q'):
            break


if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()
