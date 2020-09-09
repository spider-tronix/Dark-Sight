from cv2 import cv2
import math
import numpy as np

import os
import time

dir_path = os.path.dirname(os.path.realpath(__file__))


def nothing(nil):
    pass


cv2.namedWindow("track")
cv2.createTrackbar("sz1", "track", 15, 100, nothing)
cv2.createTrackbar("omega", "track", 95, 200, nothing)
cv2.createTrackbar("sz2", "track", 15, 100, nothing)
cv2.createTrackbar("tx", "track", 10, 100, nothing)


def DarkChannel(im, sz=cv2.getTrackbarPos("sz1", "track")):
    b, g, r = cv2.split(im)
    dc = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    dark = cv2.erode(dc, kernel)
    return dark


def AtmLight(im, dark):
    [h, w] = im.shape[:2]
    imsz = h * w
    numpx = int(max(math.floor(imsz / 1000), 1))
    darkvec = dark.reshape(imsz, 1)
    imvec = im.reshape(imsz, 3)

    indices = darkvec.argsort()
    indices = indices[imsz - numpx : :]

    atmsum = np.zeros([1, 3])
    for ind in range(1, numpx):
        atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx
    return A


def TransmissionEstimate(im, A, sz=cv2.getTrackbarPos("sz2", "track")):
    omega = cv2.getTrackbarPos("omega", "track") / 100
    im3 = np.empty(im.shape, im.dtype)

    for ind in range(0, 3):
        im3[:, :, ind] = im[:, :, ind] / A[0, ind]

    transmission = 1 - omega * DarkChannel(im3, sz)
    return transmission


def Guidedfilter(im, p, r, eps):
    mean_I = cv2.boxFilter(im, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(im * p, cv2.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(im * im, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

    q = mean_a * im + mean_b
    return q


def TransmissionRefine(im, et):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray) / 255
    r = 60
    eps = 0.0001
    t = Guidedfilter(gray, et, r, eps)

    return t


def Recover(im, t, A, tx=cv2.getTrackbarPos("tx", "track") / 100):
    res = np.empty(im.shape, im.dtype)
    t = cv2.max(t, tx)

    for ind in range(0, 3):
        res[:, :, ind] = (im[:, :, ind] - A[0, ind]) / t + A[0, ind]

    return res


def dehaze_from_file(
    src_path=os.path.join(dir_path, "img/inv.jpg"),
    op_path=os.path.join(dir_path, "img/J.jpg"),
):
    src = cv2.imread(src_path)

    I = src.astype("float64") / 255

    dark = DarkChannel(I, 15)
    A = AtmLight(I, dark)
    te = TransmissionEstimate(I, A, 15)
    t = TransmissionRefine(src, te)
    J = Recover(I, t, A, 0.1)

    cv2.imwrite(
        op_path,
        J * 255,
    )


def dehaze(src):
    I = src.astype("float32") / 255

    while 1:
        tock = time.time()
        dark = DarkChannel(I)
        A = AtmLight(I, dark)
        te = TransmissionEstimate(I, A)
        t = TransmissionRefine(src, te)
        J = Recover(I, t, A)
        deh = J * 255
        cv2.imshow("Dehazed", deh)
        op = (1 - J) * 255
        tick = time.time()
        print(1 / (tick - tock))
        # op = np.array(op,dtype= np.uint8)
        cv2.imshow("op", op)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.imwrite(os.path.join(dir_path, "img/J.jpg"), op)
            cv2.imshow("Final", cv2.imread(os.path.join(dir_path, "img/J.jpg")))
            cv2.waitKey(0)
            break


if __name__ == "__main__":
    import sys

    try:
        fn = sys.argv[1]
    except:
        fn = os.path.join(dir_path, "img/inv.jpg")

    src = cv2.imread(fn)
    src = cv2.resize(src, (480, 360))
    dehaze(src)
    # I = src.astype("float64") / 255
    #
    # dark = DarkChannel(I, 15)
    # A = AtmLight(I, dark)
    # te = TransmissionEstimate(I, A, 15)
    # t = TransmissionRefine(src, te)
    # J = Recover(I, t, A, 0.1)

    # cv2.imshow("dark",dark)
    # cv2.imshow("t",t)
    # cv2.imshow('I',src)
    # cv2.imshow('J',J)
    # cv2.imwrite(
    #     "/media/sudhar/D Drive/Spider/Dark-sight/Dark-Sight/Image Enhancement Baselines/Haze remove/img/J.jpg",
    #     J * 255,
    # )
    # cv2.waitKey()
