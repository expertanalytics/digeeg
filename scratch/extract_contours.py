import matplotlib.pyplot as plt
import numpy as np
import typing as tp

from scipy import signal

from shapely.geometry import Polygon

import cv2
import math
import itertools

from pathlib import Path

from dgimage import (
    Image,
    read_image,
    save_image,
    resample,
    get_axis,
)

from dgutils import (
    match_contours,
    filter_contours,
    remove_contours,
    filter_image,
    get_bounding_rectangle_matcher,
    get_contours,
    get_square_matcher,
    get_marker_matcher,
    get_graph_matcher,
    show,
    plot
)

from dgimage import (
    colors,
    mtableau_brg,
    mbase_brg,
    mcss_brg,
    color_to_256_BRG,
    color_to_256_RGB,
)

from dgutils import remove_structured_background


def remove_background(
    *,
    image: Image,
    smooth_kernel_size: int = 2,
    threshold_value: int = 100,
    background_kernel_size: int = 5,
) -> None:
    image.invert()              # We want the main features to be white
    image.blur(smooth_kernel_size)
    image.threshold(threshold_value)        # Removes most of the milimiter markers

    remove_structured_background(
        image_array=image.image,
        background_kernel_size=(background_kernel_size, background_kernel_size),
        smoothing_kernel_size=(smooth_kernel_size, smooth_kernel_size)
    )

    image.invert()
    image.checkpoint("preprocessed")


def markers(image: Image, blur_kernel_size=9, kernel_length=5):
    image.bgr_to_gray()
    image.invert()
    image.blur(blur_kernel_size)
    image.threshold(150)

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    horisontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))

    # vertial lines
    vertical_image = cv2.erode(image.image, vertical_kernel, iterations=4)
    cv2.dilate(vertical_image, vertical_kernel, iterations=4, dst=vertical_image)

    # Horisontal lines
    horisontal_image = cv2.erode(image.image, horisontal_kernel, iterations=4)
    cv2.dilate(image.image, horisontal_kernel, iterations=4, dst=vertical_image)

    # Compute intersection of horisontal and vertical
    cv2.bitwise_and(horisontal_image, vertical_image, dst=image.image)

    contours = get_contours(image=image)
    features = match_contours(matcher=get_marker_matcher(image=image), contours=contours)

    axis, scale = get_axis(image, features)
    image.set_axis(axis)
    image.set_scale(scale)


def extract_contours(
    *,
    image: Image,
    blur_kernel_size: int = 3,
    dilate_kernel_size: int = 3,
    num_dilate_iterations: int= 3
) -> None:
    # Remove initial guess at contours. This should leave the text blocks.
    image.invert()
    contours = get_contours(image=image)
    features = match_contours(matcher=get_graph_matcher(approximation_tolerance=1e-2), contours=contours)

    remove_contours(image, features)

    # Remove all the remaining text blobs
    image.blur(blur_kernel_size)
    image.threshold(100)
    image.morph(cv2.MORPH_DILATE, (dilate_kernel_size, dilate_kernel_size), num_dilate_iterations)

    contours = get_contours(image=image, min_size=6)

    # Compute all the bounding boxes and filter based on the aspect ratio?
    features = match_contours(
        matcher=get_bounding_rectangle_matcher(min_solidity=0.7),
        contours=contours
    )

    filter_contours(image=image, contours=features)
    image.morph(cv2.MORPH_DILATE, (dilate_kernel_size, dilate_kernel_size), num_dilate_iterations)
    image_mask = image.copy_image()

    image.reset_image("preprocessed")
    filter_image(image=image, binary_mask=image_mask == 255)
    image.checkpoint("preprocessed")

    # Match the remaining graph candidates, and remove everything else
    image.invert()
    image.blur(blur_kernel_size)
    image.threshold()
    image.morph(cv2.MORPH_DILATE, (dilate_kernel_size, dilate_kernel_size), num_dilate_iterations)

    contours = get_contours(image=image)
    features = match_contours(matcher=get_graph_matcher(), contours=contours)
    filter_contours(image=image, contours=features)

    image.reset_image("resampled")

    filter_contours(image=image, contours=features, invert=True)
    image.threshold()

    image.invert()
    contours = get_contours(image=image)
    features = match_contours(matcher=get_graph_matcher(), contours=contours)
    return features


def prepare_lines(image):
    # Reorient the image to align with axis
    markers(image)
    image.reset_image()
    resample(image, step_x=2.5e-4, step_y=2.5e-4)

    image.bgr_to_gray()
    image.checkpoint("resampled")

    remove_background(image=image, smooth_kernel_size=3)

    features = extract_contours(image=image, blur_kernel_size=3)
    return features


def plot_line(image, index, convolve = False):
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.imshow(image.image, cmap="gray")
    ax1.axvline(x=index)
    raw_signal = image.image[:, index]
    ax2.plot(raw_signal, label="raw signal")

    if convolve:
        window = signal.hann(25)
        filtered = signal.fftconvolve(raw_signal, window, mode='same') / np.sum(window)
        ax2.plot(filtered, "--", label="filtered")

    ax2.legend()
    plt.show()
    plt.close(fig)


def find_datapoints(image, start=8100):
    image.invert()
    _image = image.image
    window = signal.hann(25)
    window_sum = window.sum()

    x = np.linspace(0, 1, _image.shape[0])
    for i in range(start, _image.shape[1]):
        raw_signal = _image[:, i]
        filtered_signal = signal.fftconvolve(raw_signal, window, mode='same')/window_sum

        tmp_peaks = signal.find_peaks(filtered_signal, prominence=10)[0]
        peaks = sorted(tmp_peaks, key=lambda x: filtered_signal[x], reverse=True)[:4]

        yield i, filtered_signal[peaks]
        # fig, (ax1, ax2) = plt.subplots(2)

        # ax2.imshow(_image, cmap="gray")
        # ax2.axvline(i)

        # ax1.plot(x, raw_signal)
        # ax1.plot(x, filtered_signal, "--")
        # ax1.plot(x[peaks], filtered_signal[peaks], "x")
        # plt.show()
        # plt.close(fig)
        # return


def extract_data(image):
    print(image.image.shape)


    times = []
    data_list = []
    for t, data in find_datapoints(image, start=100):
        if data.size == 4:
            times.append(t)
            # assert len(data) == 4
            data_list.append(list(data))
        else:
            data_list.append(np.zeros(4))

    import operator

    data_arrays = [np.fromiter(map(operator.itemgetter(i), data_list), dtype=np.float_) for i in range(4)]

    fig, (ax1, ax2) = plt.subplots(2)

    ax1.imshow(image.image, cmap="gray")
    for i in range(4):
        ax2.plot(data_arrays[i] + i*500, label=f"i = {i}")
    ax2.legend()
    plt.show()


if __name__ == "__main__":
    filename = "scan4_tmp_splits/split3_scan4.png"
    image = read_image(filename)
    features = prepare_lines(image)


    tmp_image = np.ones((*image.image.shape, 3), dtype=np.uint8)
    tmp_image[:] = (255, 255, 255)      # White

    fig, ax = plt.subplots(1)
    ax.imshow(tmp_image)

    color_iterator = itertools.cycle(mtableau_brg())

    for i, c in enumerate(features):
        color = next(color_iterator)
        tmp_image = cv2.drawContours(tmp_image, features, i, color_to_256_RGB(color), cv2.FILLED)
        polygon = Polygon(c.reshape(-1, 2))
        x0, y0, x1, y1 = polygon.bounds

        tmp_image2 = np.zeros(tuple(map(math.ceil, (y1, x1))), dtype=np.uint8)
        tmp_image2 = cv2.drawContours(tmp_image2, features, i, 255, cv2.FILLED)
        np.save(f"tmp_contours/contour{i}.npy", tmp_image2)

        ann = ax.annotate(
            f"Contour {i}",
            xy=(x0, y1),
            xycoords="data",
            xytext=(0, 35),
            textcoords="offset points",
            size=10,
            bbox=dict(
                boxstyle="round",
                fc=color       # normalised color
            )
        )
    plt.imshow(tmp_image)
    plt.show()
