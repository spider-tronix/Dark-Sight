import base64

import cv2
import numpy as np
import paramiko
import zmq


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
        cv2.waitKey(1)


class SSHPi:
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

    def trigger_thermal_camera(self):
        _ = self.pi_ssh.exec_command("./test | nc 192.168.43.156 2000")

    def trigger_pi_camera(self):
        # _ = self.pi_ssh.exec_command(r'python Dark-Sight/Comms\ Over\ LAN/Video\ Capture\ Over\ TCP/server.py')
        _ = self.pi_ssh.exec_command(
            r'python /mnt/d_drive/Drive/Code/Dark-Sight/Comms Over LAN/Video Capture Over TCP/server.py')


def main():
    sshpi = SSHPi()
    sshpi.trigger_thermal_camera()
    sshpi.trigger_pi_camera()


if __name__ == '__main__':
    main()
