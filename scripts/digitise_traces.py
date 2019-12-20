import numpy as np
import typing as tp
import matplotlib.pyplot as plt
import scipy.signal as signal

import os
import logging

from pathlib import Path

from dgimage import read_image

from taylor import PointAccumulator


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


def find_datapoints(
        image: np.ndarray,
        start: int,
        num_lines: int = 1,
        show: bool = False
):
    _image = image
    window1 = signal.gaussian(50, 15)
    window1_sum = window1.sum()

    differentiator = PointAccumulator(num_lines=num_lines)

    x = np.linspace(0, 1, _image.shape[0])
    for i in range(start, _image.shape[1]):
        raw_signal = _image[:, i]
        filtered_signal = signal.fftconvolve(raw_signal, window1, mode='same')/window1_sum

        peaks = np.sort(signal.find_peaks(
            filtered_signal,
            prominence=5,
            distance=100
        )[0])

        if len(peaks) == 0:
            continue

        new_points = differentiator.add_point(i, peaks, look_back=3)

        # Probably want to move away from generator. Use differentiator always
        yield i, new_points      # TODO: Return any number of points, and use separate method to filter
        # yield i, peaks[:1]      # TODO: Return any number of points, and use separate method to filter

        if show:
            fig, (ax1, ax2) = plt.subplots(2)
            ax2.imshow(_image, cmap="gray")
            ax2.axvline(i, color="r")
            ax1.plot(raw_signal)
            ax1.plot(filtered_signal, "--")
            ax1.plot(peaks, filtered_signal[peaks], "x", linewidth=20)
            plt.show()
            plt.close(fig)


def run(
    input_image_path: Path,
    output_directory: Path,
    identfier: str,
    start_column: int = 0,
    show: bool = True
):
    trace_image = read_image(input_image_path)
 
    if len(trace_image.image.shape) > 2:
        trace_image.bgr_to_gray()
    trace_image = trace_image.image

    new_image = np.zeros(trace_image.shape)
    point_list = []
    x_list = []
    y_list = []

    for i, new_y in find_datapoints(trace_image, start=start_column):
        new_y = new_y[0]
        new_image[int(new_y), i] = 255
        x_list.append(i)
        y_list.append(int(new_y))

    x_arr = np.asarray(x_list, dtype=np.float_)
    y_arr = np.asarray(y_list, dtype=np.float_)

    y_arr -= y_arr.mean()      # mean zero
    y_arr *= -1                # flip

    if show:
        fig, (ax1, ax2) = plt.subplots(2)
        ax1.imshow(new_image)
        ax2.plot(x_arr, y_arr)
        plt.show()

    out_array = np.zeros((x_arr.size, 2))
    out_array[:, 0] = x_arr
    out_array[:, 1] = y_arr

    output_directory.mkdir(exist_ok=True, parents=True)
    np.save(output_directory / f"{identfier}_trace", out_array)


if __name__ == "__main__":
    # contours = list(np.load("contours.npy", allow_pickle=True))
    # take1(contours)
    # take2(contours)
    import sys
    input_image_path = Path(sys.argv[1])
    output_directory = Path(sys.argv[2])
    identifier = sys.argv[3]
    run(input_image_path, output_directory, identifier)
