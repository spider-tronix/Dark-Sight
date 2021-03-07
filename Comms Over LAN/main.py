from multiprocessing import Process, Array

from picam_client import PiCamera
from clahe import clahe

import socket
from time import time
import os

from cv2 import cv2
import collections
import numpy as np
from pssh.clients.native.parallel import ParallelSSHClient
import warnings

op_thermal = Array(
    "d", 24 * 32, lock=False
)  # Global variable (shared memory between threads)


class Netcat:
    """ Python 'netcat like' module """

    def __init__(self, ip, port):
        self.buff = ""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((ip, port))
        self.conn = None
        self.start = "Subpage"

    def listen(self):

        self.socket.listen(1)
        self.conn, _ = self.socket.accept()

    def read(self, length=1024):
        """ Read 1024 bytes off the socket """
        return self.conn.recv(length).decode("ascii")

    def read_until(self, data):
        """ Read data into the buffer until we have data """
        # self.buff = ""
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

        self.pi_ssh.run_command(command='tmux new-session -d "~/test_new"')
        self.pi_ssh.run_command(
            command='tmux new-session -d "bash ~/Desktop/Script/ds_startup.sh"'
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


def print_arr(arr):
    for row in arr:
        for val in row:
            print("{:4}".format(val))
        print()


def arr2heatmap(arr):

    # ax = cv2.applyColorMap( (arr * cv2.getTrackbarPos("Scale", "Trackbar")/122).astype('uint8'), cv2.COLORMAP_JET)
    # ax = cv2.applyColorMap((arr * 2.245).astype("uint8"), cv2.COLORMAP_JET)
    heatmap = cv2.normalize(arr, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    return heatmap


def thermal_process():
    global op_thermal
    nc = Netcat("192.168.0.107", 1234)
    cam = ThermalCamera()
    cam.trigger_camera()
    nc.listen()

    _ = nc.read_until("End")

    while True:
        try:
            data = nc.read_until("End")
            data = data[data.find("Subpage:") + 11 : -4]
            op_thermal[:] = list(np.concatenate(stdout2arr(data)))
            # print(op)
        except Exception as e:
            # cam.revive_cam()
            print(e)
            print("Consider restarting PI!!!!!")


def read_sensors(thermal_op_type="img", thermalimg_op_size=(24, 32)):

    Readings = collections.namedtuple("Readings", ["thermal", "normal"])

    normal_img = picam()
    thermal_readings = np.reshape(op_thermal[:], (-1, 32))

    vis = cv2.resize(thermal_readings, (thermalimg_op_size[0], thermalimg_op_size[1]))
    heatmap = arr2heatmap(vis)

    if thermal_op_type == "img":
        reading = Readings(heatmap, normal_img)
    else:
        reading = Readings(thermal_readings, normal_img)

    return reading


def main():

    while True:
        tick = time()
        reading = read_sensors(thermalimg_op_size=(480, 360))

        try:
            cv2.imshow("Thermal", reading.thermal)
            cv2.imshow("Normal", reading.normal)

            # Processing the camera readings
            op = clahe(reading.normal)
            cv2.imshow("Output", op)

            hsvImg = cv2.cvtColor(reading.normal, cv2.COLOR_BGR2HSV)
            # decreasing the V channel by a factor from the original
            hsvImg[..., 2] = hsvImg[..., 2] * 0.2
            cv2.imshow("Darker Input", cv2.cvtColor(hsvImg, cv2.COLOR_HSV2RGB))

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Exiting...")
                output_path = os.path.dirname(__file__)
                cv2.imwrite(os.path.join(output_path, "Thermal image.jpg"), reading.thermal)
                cv2.imwrite(os.path.join(output_path, "Normal image.jpg"), reading.normal)
                cv2.imwrite(os.path.join(output_path, "Output image.jpg"), op)
                cv2.imwrite(
                    os.path.join(output_path, "Darker Normal image.jpg"),
                    cv2.cvtColor(hsvImg, cv2.COLOR_HSV2RGB),
                )
                break
            tock = time()
            print("FPS: ", 1 / (tock - tick))
        except:
            pass

    picam.release()
    cv2.destroyAllWindows()
    p.terminate()


def nothing(nil):
    pass


if __name__ == "__main__":

    warnings.filterwarnings("ignore")

    p = Process(target=thermal_process)
    p.start()

    picam = PiCamera()

    main()
