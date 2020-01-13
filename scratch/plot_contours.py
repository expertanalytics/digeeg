import numpy as np
import typing as tp
import matplotlib.pyplot as plt

import pickle

import scipy.signal as signal

import shapely.geometry

import scipy.interpolate as interp

from taylor import PointAccumulator

from dataclasses import dataclass


def find_datapoints(image, start=0):
    # _image = 255 - image
    _image = image
    window1 = signal.gaussian(50, 15)
    window1_sum = window1.sum()

    differentiator = PointAccumulator(num_lines=1)

    x = np.linspace(0, 1, _image.shape[0])
    for i in range(start, _image.shape[1]):
        raw_signal = _image[:, i]
        filtered_signal = signal.fftconvolve(raw_signal, window1, mode='same')/window1_sum

        peaks = np.sort(signal.find_peaks(
            filtered_signal,
            prominence=5,
            distance=100
        )[0])

        # peaks = sorted(tmp_peaks, key=lambda x: filtered_signal[x], reverse=True)[:4]

        # yield i, filtered_signal[peaks]
        if len(peaks) == 0:
            continue

        new_points = differentiator.add_point(i, peaks, look_back=3)

        # Probably want to move away from generator. Use differentiator always
        yield i, new_points      # TODO: Return any number of points, and use separate method to filter
        # yield i, peaks[:1]      # TODO: Return any number of points, and use separate method to filter

        fig, (ax1, ax2) = plt.subplots(2)
        ax2.imshow(_image, cmap="gray")
        ax2.axvline(i, color="r")

        ax1.plot(raw_signal)
        ax1.plot(filtered_signal, "--")
        ax1.plot(peaks, filtered_signal[peaks], "x", linewidth=20)
        plt.show()
        plt.close(fig)


if __name__ == "__main__":
    # contours = list(np.load("contours.npy", allow_pickle=True))
    # take1(contours)
    # take2(contours)

    for contour_number in [3]:
        contour_image = np.load(f"tmp_contours/image_contour{contour_number}.npy")
        # plt.imshow(contour_image)
        # plt.show()
        # assert False
        # print(contour_image.shape)

        new_image = np.zeros(contour_image.shape)
        point_list = []
        x_list = []
        y_list = []
        for i, new_y in find_datapoints(contour_image, start=7300):
            # point_list.append((i, new_y))
            new_y = new_y[0]
            new_image[int(new_y), i] = 255
            x_list.append(i)
            y_list.append(int(new_y))

        fig, (ax1, ax2) = plt.subplots(2)
        ax1.imshow(new_image)

        x_arr = np.asarray(x_list, dtype=np.float_)
        y_arr = np.asarray(y_list, dtype=np.float_)

        y_arr -= y_arr.mean()      # mean zero
        y_arr *= -1                # flip

        ax2.plot(x_arr, y_arr)
        out_array = np.zeros((x_arr.size, 2))
        out_array[:, 0] = x_arr
        out_array[:, 1] = y_arr
        np.save(f"tmp_lines/out_array{contour_number}", out_array)
        plt.show()

        # from scipy.signal import welch
        # f, pxx = welch(y_arr, 1600e3)

        # plt.loglog(f, pxx)
        # plt.show()

        # for i in range(100, contour_image.shape[1]):
        # for i in range(100, 200):
        #     print(np.median(contour_image[i, :]))
