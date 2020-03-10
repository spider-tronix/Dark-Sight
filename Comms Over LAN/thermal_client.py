import socket
import time

import numpy as np


class Netcat:
    """ Python 'netcat like' module """

    def __init__(self, ip, port):
        self.buff = ""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((ip, port))

        self.socket.listen(2)
        self.conn, addr = self.socket.accept()

    def read(self, length=1024):
        """ Read 1024 bytes off the socket """

        return self.conn.recv(length)

    def read_until(self, data):
        """ Read data into the buffer until we have data """

        while not data in self.buff:
            self.buff += self.conn.recv(1024).decode('ascii')

        pos = self.buff.find(data)
        rval = self.buff[:pos + len(data)]
        self.buff = self.buff[pos + len(data):]

        return rval

    def write(self, data):
        self.socket.send(data)

    def close(self):
        self.socket.close()


# start a new Netcat() instance
nc = Netcat('192.168.43.156', 2000)


def stdout2arr(string):
    tem_row = list(string.strip().split('\n'))
    temp = []
    for col in tem_row:
        temp.append(list(map(float, col.strip().split(' '))))
    temp = np.array(temp)

    temp = np.flipud(temp)
    temp = np.fliplr(temp)
    return temp


def main():
    _ = nc.read_until('End')
    while True:
        tick = time.time()
        data = nc.read_until('End')
        data = data[data.find('Subpage:') + 11:-4]
        proc = stdout2arr(data)
        tock = time.time()
        print(-(tick - tock))


if __name__ == '__main__':
    main()
