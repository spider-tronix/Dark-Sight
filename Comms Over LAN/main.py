import base64
import socket
import time

import cv2
import numpy as np
import paramiko
import ray
import zmq

ray.init()


@ray.remote
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
        return source


class Netcat:
    """ Python 'netcat like' module """
    socket_args = {'family': socket.AF_INET,
                   'type': socket.SOCK_STREAM}

    def __init__(self, ip, port):
        self.buff = ""
        self.socket = socket.socket(**self.socket_args)
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
        self.conn.close()


def custom_serializer(obj):
    obj.close()
    return obj.socket_args


def custom_deserializer(value):
    obj = Netcat('192.168.43.156', 2000)
    obj.listen()
    return obj


ray.register_custom_serializer(
    Netcat, serializer=custom_serializer, deserializer=custom_deserializer)


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
        # self.pi_ssh = ParallelSSHClient(hosts=[self.ip], user=self.uname, password=self.passd)
        self.pi_ssh = paramiko.SSHClient()
        self.connect_ssh()

    def connect_ssh(self):
        self.pi_ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.pi_ssh.connect(hostname=self.ip, username=self.uname, password=self.passd)

    def trigger_thermal_camera(self):
        # self.pi_ssh.run_command(command='tmux new-session -d "while true; do (~/test | nc 192.168.43.156 2000); done"')
        # self.pi_ssh.run_command(command='while true; do (~/test | nc 192.168.43.156 2000); done')
        _, _, err = self.pi_ssh.exec_command(
            command='tmux send -t 0 "while true; do (~/test | nc 192.168.43.156 2000); done" ENTER',
            get_pty=True)
        print(f"Errors : {err.readlines()}")
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


@ray.remote
def read_proc_thermal(nc):
    tick = time.time()

    data = nc.read_until('End')
    print(data)
    data = data[data.find('Subpage:') + 11:-4]
    proc = stdout2arr(data)

    tock = time.time()
    print(-(tick - tock))

    vis = cv2.resize(proc, (960, 720))
    heatmap = arr2heatmap(vis)
    return heatmap


def main():
    picam_recv = PiCam.remote()
    sshpi = SSHPi()
    nc = Netcat('192.168.43.156', 2000)

    # sshpi.trigger_pi_camera()
    sshpi.trigger_thermal_camera()

    nc.listen()
    one = nc.read_until('End')
    print(one)
    nc.socket.close()
    thermal = 'MLX90640 Feed'
    cam = 'PiCam Feed'
    cv2.namedWindow(thermal, cv2.WINDOW_NORMAL)

    while True:
        try:
            # img = picam_recv.recv.remote()
            heatmap = read_proc_thermal.remote(nc)
            # img, heatmap = ray.get([img, heatmap])
            # img = ray.get(img)
            heatmap = ray.get(heatmap)
            cv2.imshow(thermal, heatmap)
            # cv2.imshow(cam, img)
            ch = cv2.waitKey(1)
            if ch == ord('q'):
                break

        except Exception as e:
            print(e)

    cv2.destroyAllWindows()
    sshpi.tmux_kill()


if __name__ == '__main__':
    main()
