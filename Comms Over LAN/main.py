import base64

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


# @ray.remote
class SSHPi:
    def __init__(self):
        self.port = '22'
        self.uname = 'pi'
        self.passd = 'sharan'
        self.ip = '192.168.43.38'
        # self.pi_ssh = paramiko.SSHClient()
        # self.connect_ssh()
        self.pi_ssh = ParallelSSHClient(hosts=[self.ip], user=self.uname, password=self.passd)

    def connect_ssh(self):
        # self.pi_ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        # self.pi_ssh.connect(hostname=self.ip, username=self.uname, password=self.passd)
        pass

    def trigger_thermal_camera(self):
        _ = self.pi_ssh.exec_command("./test | nc 192.168.43.156 2000")

    def trigger_pi_camera(self):
        # _, _, err = self.pi_ssh.exec_command(
        #     r'~/env/bin/python3 Dark-Sight/Comms\ Over\ LAN/Video\ Capture\ Over\ TCP/server.py')
        self.pi_ssh.run_command(
            command=r'tmux new-session -d '
                    r'"~/env/bin/python3 Dark-Sight/Comms\ Over\ LAN/Video\ Capture\ Over\ TCP/server.py"')
        print("PiCam is Live!")

    def tmux_kill(self):
        self.pi_ssh.run_command(
            command=r'tmux kill-server')


def main():
    # sshpi = SSHPi.remote()
    picam_recv = PiCam()
    # sshpi.trigger_pi_camera.remote()
    # ray.get([
    #     # sshpi.trigger_thermal_camera().remote(),
    #     sshpi.trigger_pi_camera.remote()
    # ])
    sshpi = SSHPi()
    sshpi.trigger_pi_camera()
    while True:
        ch = picam_recv.recv()
        if ch == ord('q'):
            break
    cv2.destroyAllWindows()
    sshpi.tmux_kill()


if __name__ == '__main__':
    main()
