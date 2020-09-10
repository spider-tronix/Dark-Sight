from cv2 import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

current_dir = (
    "/media/sudhar/D Drive/Spider/Dark-sight/Dark-Sight/Image Enhancement Baselines"
)


def nothing(sdg):
    pass


def clahe_tune(img):
    """
    Apply CLACHE algorithm to image with trackbars to tune
    :param img: Input image of RGB format
    :output: filtered image
    """
    res = []
    tg = cv2.getTrackbarPos("tg", "track") + 1
    clahe = cv2.createCLAHE(
        clipLimit=cv2.getTrackbarPos("clip", "track"), tileGridSize=(tg, tg)
    )
    for channel in np.dsplit(img, img.shape[-1]):
        res.append(clahe.apply(channel))
    res = np.dstack(res)

    k = 2 * cv2.getTrackbarPos("kernel", "track") - 1
    t = cv2.getTrackbarPos("t", "track")

    res = cv2.GaussianBlur(res, (k, k), 0)  # Apply filter
    res = cv2.bilateralFilter(res, t, 75, 75)

    cv2.imshow("result", res)
    return res


def clahe(img, disp=False):
    """
    Apply CLACHE algorithm to image
    :param img: Input image of RGB format
    :param disp: Control display flag
    :output: filtered image
    """
    res = []
    clahe = cv2.createCLAHE(18, tileGridSize=(21, 21))
    for channel in np.dsplit(img, img.shape[-1]):
        res.append(clahe.apply(channel))
    res = np.dstack(res)

    k = 5
    t = 5
    res = cv2.GaussianBlur(res, (k, k), 0)  # Apply filter
    res = cv2.bilateralFilter(res, t, 75, 75)

    if disp:
        cv2.imshow("result", res)
    return res


def main():
    global short
    while 1:
        k = clahe_tune(short)

        if cv2.waitKey(1) & 0xFF == ord("q"):

            cv2.imwrite("Final.jpg", k)
            break


if __name__ == "__main__":

    cv2.namedWindow("track")
    cv2.createTrackbar("clip", "track", 18, 100, nothing)
    cv2.createTrackbar("tg", "track", 20, 200, nothing)
    cv2.createTrackbar("kernel", "track", 3, 200, nothing)
    cv2.createTrackbar("t", "track", 5, 200, nothing)

    dataset_no = 2

    short = cv2.imread(
        os.path.join(
            current_dir,
            os.path.join("dataset/" + str(dataset_no), str(dataset_no) + ".JPG"),
        )
    )
    long = cv2.imread(
        os.path.join(
            current_dir,
            os.path.join("dataset/" + str(dataset_no), str(dataset_no) + "_high.JPG"),
        )
    )

    # short = cv2.resize(short, (480, 360))
    # long = cv2.resize(long, (480, 360))

    cv2.imshow("Lit", long)
    main()
