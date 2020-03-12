import base64
import socket
import time

import cv2
import numpy as np
import ray
import zmq
from pssh.clients.native.parallel import ParallelSSHClient

ray.init()


class PiCam:
    def __init__(self):
        context = zmq.Context()
        self.footage_socket = context.socket(zmq.SUB)
        self.footage_socket.bind('tcp://*:5555')
        self.footage_socket.setsockopt_string(zmq.SUBSCRIBE, np.unicode(''))

    def recv(self):
        frame = self.footage_socket.recv_string()
        img = base64.b64decode(frame)
        npimg = np.fromstring(img, dtype=np.uint8)
        source = cv2.imdecode(npimg, 1)
        cv2.imshow("Stream", source)
        ch = cv2.waitKey(1)
        return ch


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


def stdout2arr(string):
    tem_row = list(string.strip().split('\n'))
    temp = []
    for col in tem_row:
        temp.append(list(map(float, col.strip().split(' '))))
    temp = np.array(temp, dtype=np.float32)

    # temp = np.flipud(temp)
    temp = np.rot90(temp)
    temp = np.fliplr(temp)
    return temp


def arr2heatmap(arr):
    heatmap = cv2.normalize(arr, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return heatmap


# @ray.remote
class SSHPi:
    def __init__(self):
        self.port = '22'
        self.uname = 'pi'
        self.passd = 'sharan'
        self.ip = '192.168.43.38'
        self.pi_ssh = ParallelSSHClient(hosts=[self.ip], user=self.uname, password=self.passd)

    def trigger_thermal_camera(self):
        self.pi_ssh.run_command(command='tmux new-session -d "~/test | nc 192.168.43.156 2000"')
        print("Thermal camera is Live!")

    def trigger_pi_camera(self):
        self.pi_ssh.run_command(
            command=r'tmux new-session -d '
                    r'"~/env/bin/python3 Dark-Sight/Comms\ Over\ LAN/Video\ Capture\ Over\ TCP/server.py"')
        print("PiCam is Live!")

    def tmux_kill(self):
        self.pi_ssh.run_command(
            command=r'tmux kill-server')
        print("Tmux server killed :(")


def main():
    picam_recv = PiCam()
    sshpi = SSHPi()
    nc = Netcat('192.168.43.156', 2000)

    sshpi.trigger_pi_camera()
    sshpi.trigger_thermal_camera()

    nc.listen()
    _ = nc.read_until('End')
    thermal = 'Thermal Feed'
    cv2.namedWindow(thermal, cv2.WINDOW_NORMAL)

    while True:
        try:
            tick = time.time()

            data = nc.read_until('End')
            data = data[data.find('Subpage:') + 11:-4]
            proc = stdout2arr(data)

            tock = time.time()
            print(-(tick - tock))

            vis = cv2.resize(proc, (960, 720))
            heatmap = arr2heatmap(vis)
            ch = picam_recv.recv()

            cv2.imshow(thermal, heatmap)
            ch = cv2.waitKey(1)

            if ch == ord('q'):
                break

        except Exception as e:
            print(e)

    cv2.destroyAllWindows()
    sshpi.tmux_kill()


if __name__ == '__main__':
    main()

"""
    # sshpi = SSHPi.remote()
    # sshpi.trigger_pi_camera.remote()
    # ray.get([
    #     # sshpi.trigger_thermal_camera().remote(),
    #     sshpi.trigger_pi_camera.remote()
    # ])
"""
