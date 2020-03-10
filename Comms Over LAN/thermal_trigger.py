import matplotlib.pyplot as plt
import numpy as np
import paramiko


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
        stdin, stdout, stderr = self.pi_ssh.exec_command(
            "(sleep 2; echo a) | '/home/pi/darkSight/Dark-Sight/Thermal Camera Libs/pimoroni/examples/fbuf'")
        print(stdout.read())
        self.txt2jpg()

    @staticmethod
    def txt2jpg():
        file = open('/home/syzygianinfern0/sambashare/timestamp.txt', 'r')
        filename = file.read().strip()
        file.close()
        file = open('/home/syzygianinfern0/sambashare/' + filename + '/jpg_temp.txt', 'r')

        while (not file):
            file = open('/home/syzygianinfern0/sambashare/' + filename + '/jpg_temp.txt', 'r')

        string = ""
        for each in file:
            string += each
        color = list(string.strip().split('#'))
        ir = list(color[0].strip().split('\n'))
        ig = list(color[1].strip().split('\n'))
        ib = list(color[2].strip().split('\n'))
        irr = []
        igg = []
        ibb = []
        img = []
        for r in ir:
            irr.append(list(map(int, r.strip().split('\t'))))
        for g in ig:
            igg.append(list(map(int, g.strip().split('\t'))))
        for b in ib:
            ibb.append(list(map(int, b.strip().split('\t'))))

        img.append(irr)
        img.append(igg)
        img.append(ibb)

        img = np.array(img)

        img = img.transpose((1, 2, 0))
        img = np.flipud(img)
        img = np.fliplr(img)

        print(img.shape)
        img = img / 255
        direc = '/home/syzygianinfern0/sambashare/' + filename

        plt.imsave(direc + '/jpg_temp.jpg', img)
        file.close()

        file = open('/home/syzygianinfern0/sambashare/' + filename + '/temp.txt', 'r')

        string = ""
        for each in file:
            string += each
        tem_row = list(string.strip().split('\n'))
        temp = []
        for col in tem_row:
            temp.append(list(map(float, col.strip().split('\t'))))
        temp = np.array(temp)

        temp = np.flipud(temp)
        temp = np.fliplr(temp)

        temp = temp / 255
        plt.imsave(direc + '/temp.jpg', temp)


def main():
    camera = ThermalCamera()
    camera.trigger_camera()
