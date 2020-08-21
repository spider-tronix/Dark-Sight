from multiprocessing import Process, Array

import picam_client as picam

import socket
import time

from cv2 import cv2
import collections
import numpy as np
from pssh.clients.native.parallel import ParallelSSHClient

op = Array("d", 24 * 32, lock=False)  # Global variable (shared memory between threads)


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
            self.buff += self.conn.recv(1024).decode("ascii")

        pos = self.buff.find(data)
        rval = self.buff[: pos + len(data)]
        self.buff = self.buff[pos + len(data) :]

        return rval

    def write(self, data):
        self.socket.send(data)

    def close(self):
        self.socket.close()


class ThermalCamera:
    def __init__(self):
        self.port = "22"
        self.uname = "pi"
        self.passd = "ni6ga2rd"
        self.ip = "192.168.0.109"
        self.pi_ssh = ParallelSSHClient(
            hosts=[self.ip], user=self.uname, password=self.passd
        )
        self.connect_ssh()

    def connect_ssh(self):
        # self.pi_ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        # self.pi_ssh.connect(hostname=self.ip, username=self.uname, password=self.passd)
        pass

    def trigger_camera(self):
        # _, _, err = self.pi_ssh.exec_command(
        #     'tmux attach-session -d "~/test | nc 192.168.43.156 2000"', get_pty=True)
        # _, _, err = self.pi_ssh.exec_command('tmux send -t one "~/test | nc 192.168.43.156 2000" ENTER', get_pty=True)
        # _, _, err = self.pi_ssh.exec_command('tmux send -t two "timeout 4 ~/bin/fbuf" ENTER', get_pty=True)
        # _ = self.pi_ssh.exec_command('tmux new-session -d "timeout 4 ~/bin/fbuf"')

        self.pi_ssh.run_command(
            command='tmux new-session -d "~/test | nc 192.168.0.104 2000"'
        )

        # self.revive_cam()
        # err = err.readlines()
        # print(f"Errors on cmd : {''.join(err)}")

    def revive_cam(self):
        # self.pi_ssh.run_command('tmux new-session -d "timeout 3 fbuf"')
        self.pi_ssh.run_command('tmux new-session -d "fbuf"')


def stdout2arr(string):
    tem_row = list(string.strip().split("\n"))
    temp = []
    for col in tem_row:
        temp.append(list(map(float, col.strip().split(" "))))
    temp = np.array(temp, dtype=np.float32)

    # temp = np.flipud(temp)
    temp = np.rot90(temp)
    temp = np.fliplr(temp)
    return temp


def arr2heatmap(arr):
    heatmap = cv2.normalize(
        arr, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return heatmap


def thermal_process():
    global op
    nc = Netcat("192.168.0.104", 2000)
    cam = ThermalCamera()
    cam.trigger_camera()

    nc.listen()
    _ = nc.read_until("End")

    while True:
        try:
            data = nc.read_until("End")
            data = data[data.find("Subpage:") + 11 : -4]
            op[:] = list(np.concatenate(stdout2arr(data)))
            # print(op)
        except Exception as e:
            print(e)


def read_sensors(thermal_op_type="img", thermalimg_op_size=(24, 32)):

    Readings = collections.namedtuple("Readings", ["thermal", "normal"])

    thermal_readings = np.reshape(op[:], (-1, 32))
    normal_img = picam.pi_img(cap)

    vis = cv2.resize(thermal_readings, (thermalimg_op_size[0], thermalimg_op_size[1]))
    heatmap = arr2heatmap(vis)

    if thermal_op_type == "img":
        reading = Readings(heatmap, normal_img)
    else:
        reading = Readings(thermal_readings, normal_img)

    return reading


def main():

    while True:
        reading = read_sensors(thermalimg_op_size=(480, 360))

        cv2.imshow("thermal", reading.thermal)
        cv2.imshow("image", reading.normal)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    cap = picam.initialize()

    p = Process(target=thermal_process)
    p.start()

    main()

