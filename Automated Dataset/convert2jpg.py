# OM NAMO NARAYANA

import matplotlib.pyplot as plt
import numpy as np


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
    # print(ir)
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

    img = np.array([np.array(img[0]).T, np.array(img[1]).T, np.array(img[2]).T]).T
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
    temp = temp / 255
    plt.imsave(direc + '/temp.jpg', temp)
