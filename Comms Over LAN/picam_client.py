import base64
import socket
import time

from cv2 import cv2
import numpy as np
import paramiko
import ray
import zmq

class SSHPi:
    def __init__(self):
        self.port = '22'
        self.uname = 'pi'
        self.passd = 'ni6ga2rd'
        self.ip = '192.168.0.109'
        # self.pi_ssh = ParallelSSHClient(hosts=[self.ip], user=self.uname, password=self.passd)
        self.pi_ssh = paramiko.SSHClient()
        self.connect_ssh()

    def connect_ssh(self):
        self.pi_ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.pi_ssh.connect(hostname=self.ip, username=self.uname, password=self.passd)

    def trigger_thermal_camera(self):
        self.pi_ssh.exec_command(
            command='tmux new-session -d " ~/test | (while true; do (nc 192.168.0.104 2000); done) "')
        # self.pi_ssh.exec_command(command='tmux new-session -d "~/test | nc 192.168.43.156 2000"')
        # self.pi_ssh.run_command(command='while true; do (~/test | nc 192.168.43.156 2000); done')
        # _, _, err = self.pi_ssh.exec_command(
        #     command='tmux send -t 0 "while true; do (~/test | nc 192.168.43.156 2000); done" ENTER',
        #     get_pty=True)
        # print(f"Errors : {err.readlines()}")
        print("Thermal camera is Live!")

    def trigger_pi_camera(self):
        self.pi_ssh.exec_command(
            command=r'tmux new-session -d '
                    r'"raspivid -t 0 -b 2000000 -fps 10 -w 800 -h 600 -o - | nc -p 1904 -u 192.168.0.104 1234 "') # Port: 1234 demux=h264
        print("PiCam is Live!")

    def tmux_kill(self):
        self.pi_ssh.exec_command(
            command=r'tmux kill-server')
        print("Tmux server killed :(")

class Netcat:
    """ Python 'netcat like' module """
    socket_args = {'family': socket.AF_INET,
                   'type': socket.SOCK_DGRAM}
    ip = None
    port = None
    def __init__(self, ip, port):
        self.buff = ""
        self.ip = ip
        self.port = port
        # self.socket = socket.socket(**self.socket_args)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((ip, port))
        self.conn = None

    def listen(self):
        self.socket.listen(2)
        self.conn, addr = self.socket.accept()

    def read(self, length=1024):
        """ Read 1024 bytes off the socket """
        return self.socket.recvfrom(self.port)

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

def main():
    print("Waiting for PI-Camera stream")
    nc = Netcat('192.168.0.104', 1234)
    while(1):
        d = nc.read()
        img = base64.b64decode(d)
        npimg = np.fromstring(img, dtype=np.uint8)
        source = cv2.imdecode(npimg, 1)
        cv2.imshow('this',source)

if __name__ == '__main__':
    main()