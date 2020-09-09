from cv2 import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from scipy import ndimage


class HistogramEqaulize:
    def plot_hist(self, img, disp=False):
        """
        Plots the histogram of an RGB image in all the channels

        :param img: Input image
        """
        color = ("b", "g", "r")
        for i, col in enumerate(color):
            histr = cv2.calcHist([img], [i], None, [256], [0, 256])
            plt.plot(histr, color=col)
            plt.xlim([0, 256])
        if disp:
            plt.show()

    def equalize_hist(self, img, disp=False):
        """
        Runs the equalizeHist over each channel and cats them
        for output

        :param img: Input image of RGB format
        """
        res = []
        for channel in np.dsplit(img, img.shape[-1]):
            res.append(cv2.equalizeHist(channel))
        res = np.dstack(res)
        if disp:
            cv2.imshow("Histogram Equalized Image", res)
        return res

    def denoise(self, img, disp=False):
        dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        if disp:
            cv2.imshow("Denoised", dst)
        return dst

    def scfilter(self, image, iterations=1, kernel=2, disp=False):
        """
        Sine‐cosine filter.
        kernel can be tuple or single value.
        Returns filtered image.
        """
        image = np.array(image, dtype=np.int64)
        for n in range(iterations):
            image = np.arctan2(
                ndimage.filters.uniform_filter(np.sin(image), size=kernel),
                ndimage.filters.uniform_filter(np.cos(image), size=kernel),
            )
        if disp:
            cv2.imshow("Final", image)
        return image


def main(short_exposure, long_exposure):
    hist = HistogramEqaulize()

    short = cv2.imread(short_exposure)
    long = cv2.imread(long_exposure)

    short = cv2.resize(short, (480, 360))
    long = cv2.resize(long, (480, 360))

    hist.plot_hist(short)
    hist.plot_hist(long)

    short_eq = hist.equalize_hist(short, disp=True)
    hist.plot_hist(short_eq)

    long_eq = hist.equalize_hist(long)
    hist.plot_hist(long_eq)

    # hist.denoise(short_eq, disp=True)
    op = hist.scfilter(short_eq, disp=True)
    print("Press any key to continue!")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.realpath(__file__))

    dataset_no = 3

    img_path_dim = os.path.join(
        current_dir,
        os.path.join("dataset/" + str(dataset_no), str(dataset_no) + ".JPG"),
    )
    img_path_lit = os.path.join(
        current_dir,
        os.path.join("dataset/" + str(dataset_no), str(dataset_no) + "_high.JPG"),
    )

    main(img_path_dim, img_path_lit)
