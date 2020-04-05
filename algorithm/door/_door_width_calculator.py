import os

import numpy as np
import matplotlib.pyplot as plt

from skimage.filters import gaussian, threshold_sauvola
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks


class DoorWidthCalculator(object):
    @staticmethod
    def find_door_width(data_img, hyperparameters, logger=None, img_name=None, output_dir=None, verbose=True):
        vertical_line_coeff = hyperparameters["vertical_line_hough_coeff"]

        data_img_blur = gaussian(data_img, sigma=3.5, multichannel=True)
        data_img_blur_gray = rgb2gray(data_img_blur)

        data_img_filtered = data_img_blur_gray <= threshold_sauvola(data_img_blur_gray)

        h, theta, d = hough_line(data_img_filtered)

        if verbose:
            fig, ax = plt.subplots(1, 2, figsize=(15, 6))
            ax[0].imshow(data_img_filtered, cmap="gray")
            ax[0].set_title('Input image')
            ax[0].set_axis_off()

            ax[1].imshow(data_img_filtered, cmap="gray")

        x_min = data_img.shape[1]
        x_max = 0.0
        for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
            y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
            y1 = (dist - data_img_filtered.shape[1] * np.cos(angle)) / np.sin(angle)
            if np.abs(angle) <= vertical_line_coeff:
                if x_min > dist:
                    x_min = dist
                if x_max < dist:
                    x_max = dist

            if verbose:
                ax[1].plot((0, data_img_filtered.shape[1]), (y0, y1), '-r')

        if verbose:
            ax[1].set_xlim((0, data_img_filtered.shape[1]))
            ax[1].set_ylim((data_img_filtered.shape[0], 0))
            ax[1].set_axis_off()
            ax[1].set_title('Detected lines\nThreshold: threshold sauvola')

            path_to_save_fig = os.path.normpath(os.path.join(output_dir, "door_hough_%s.png" % img_name))
            plt.tight_layout()
            fig.savefig(path_to_save_fig)
            plt.close()
            logger.info("Results of Hough transform saved by path: %s" % path_to_save_fig)

        width = (x_max - x_min) / data_img.shape[1]
        if verbose:
            logger.info("Found width of door: %.4f" % width)

        return width
