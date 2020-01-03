import matplotlib.pyplot as plt
import numpy as np
import typing as tp

from scipy import signal

from shapely.geometry import Polygon

import cv2
import math
import itertools
import os
import logging

from pathlib import Path

from dgimage import (
    Image,
    read_image,
    save_image,
    resample,
    get_axis,
    load_image
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
    plot,
    save,
    get_debug_path,
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


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


def remove_background(
    *,
    image: Image,
    smooth_kernel_size: int = 2,
    threshold_value: int = 100,
    background_kernel_size: int = 5,
    debug: bool = False
) -> None:
    """Remove the background milimeter pattern."""
    if debug:
        debug_path = get_debug_path("remove_background")
        save(np.ndarray, debug_path, "input")

    image.invert()              # We want the main features to be white
    image.blur(smooth_kernel_size)
    image.threshold(threshold_value)        # Removes most of the milimiter markers

    remove_structured_background(
        image_array=image.image,
        background_kernel_size=(background_kernel_size, background_kernel_size),
        smoothing_kernel_size=(smooth_kernel_size, smooth_kernel_size),
        debug=debug
    )
    if debug:
        save(np.ndarray, debug_path, "output")

    image.invert()
    image.checkpoint("preprocessed")


def extract_contours(
    *,
    image: Image,
    blur_kernel_size: int = 3,
    dilate_kernel_size: int = 3,
    num_dilate_iterations: int= 3,
    debug: bool = True
) -> tp.List[np.ndarray]:
    """Segment the EEG traces.

    Use a series of convolutions, filtering and edge tracking to extract the contours.

    blur_kernel_size:
        Used in conjunction with thresholding to binarise the image.
    dilate_kernel_size:
        Dilation is used with filtering to be aggressive in removing elements.
    num_dilate_iterations:
        THe number of dilate iterations.
    """
    if debug:
        debug_path = get_debug_path("extract_contours")
        save(np.ndarray, debug_path, "input")
    # Remove initial guess at contours. This should leave the text blocks.
    image.invert()
    contours = get_contours(image=image)
    features = match_contours(matcher=get_graph_matcher(approximation_tolerance=1e-2), contours=contours)
    remove_contours(image, features)
    if debug:
        save(np.ndarray, debug_path, "remove_contours")

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
    filter_contours(image_array=image.image, contours=features)
    image.morph(cv2.MORPH_DILATE, (dilate_kernel_size, dilate_kernel_size), num_dilate_iterations)
    image_mask = image.copy_image()

    image.reset_image("preprocessed")
    filter_image(image_array=image.image, binary_mask=image_mask == 255)
    image.checkpoint("preprocessed")
    if debug:
        save(np.ndarray, debug_path, "filter_contours1")

    # Match the remaining graph candidates, and remove everything else
    image.invert()
    image.blur(blur_kernel_size)
    image.threshold()
    image.morph(cv2.MORPH_DILATE, (dilate_kernel_size, dilate_kernel_size), num_dilate_iterations)

    contours = get_contours(image=image)
    features = match_contours(matcher=get_graph_matcher(), contours=contours)
    filter_contours(image_array=image.image, contours=features)
    if debug:
        save(np.ndarray, debug_path, "filter_contours2")

    image.reset_image("resampled")

    # TODO: Why invert? Has something to do with the fill color in filter_contours
    image.invert()
    filter_contours(image_array=image.image, contours=features)
    image.invert()
    if debug:
        save(np.ndarray, debug_path, "filter_contours3")

    image.blur(blur_kernel_size)
    image.threshold(100)

    image.invert()
    contours = get_contours(image=image)
    features = match_contours(matcher=get_graph_matcher(), contours=contours)
    return features


def prepare_lines(image):
    if debug:
        debug_path = get_debug_path("prepare_lines")
        save(np.ndarray, debug_path, "input")

    image.bgr_to_gray()
    image.checkpoint("resampled")

    # TODO: I can move remove_background in here
    remove_background(image=image, smooth_kernel_size=3)
    features = extract_contours(image=image, blur_kernel_size=3)
    if debug:
        save(np.ndarray, debug_path, "remove_background")

    image.invert()
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

        # TODO: How can I plot this in a sensible manner? Require docker with X-forwarding?
        # fig, (ax1, ax2) = plt.subplots(2)
        # ax2.imshow(_image, cmap="gray")
        # ax2.axvline(i)

        # ax1.plot(x, raw_signal)
        # ax1.plot(x, filtered_signal, "--")
        # ax1.plot(x[peaks], filtered_signal[peaks], "x")
        # plt.show()
        # plt.close(fig)
        return


def extract_data(image):
    print(image.image.shape)

    times = []
    data_list = []
    for t, data in find_datapoints(image, start=100):
        if data.size == 4:
            times.append(t)
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


def run(
    input_image_path: Path,
    output_directory: Path,
    identifier: str,
    scale: float = None,
    debug:bool
):
    """Remove the background, segment the contours and save the segmented lines as pngs."""
    image = read_image(input_image_path)
    if debug:
        debug_path = get_debug_path("prepare_lines")
        save(np.ndarray, debug_path, "input")

    image.bgr_to_gray()
    image.checkpoint("resampled")

    remove_background(image=image, smooth_kernel_size=3)
    features = extract_contours(image=image, blur_kernel_size=3)
    if debug:
        save(np.ndarray, debug_path, "remove_background")
    image.invert()


    image.reset_image("resampled")
    image.invert()

    output_directory.mkdir(exist_ok=True, parents=True)

    for i, c in enumerate(features):
        tmp_image = image.copy_image()
        filter_contours(image_array=tmp_image, contours=[c])
        clipped_contour = tmp_image[~np.all(tmp_image == 0, axis=1)]
        save_image(output_directory / f"{identifier}_trace{i}.png", Image(clipped_contour))

    ##################
    ### Make image ###
    ##################

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

    ax.imshow(tmp_image)
    ax.set_title("A digitised paper strip")
    if scale is not None:       # multiply by distance between black sqaures?
        ax.set_xticklabels(["{:.1f} cm".format(15*i/scale) for i in ax.get_xticks()])
        ax.set_yticklabels(["{:.1f} cm".format(15*i/scale) for i in ax.get_yticks()])
    ax.set_ylabel("Voltage")
    ax.set_xlabel("Time")
    fig.savefig(output_directory / f"{identifier}_annotated.png")


if __name__ == "__main__":
    import sys
    input_image_path = Path(sys.argv[1])
    output_directory = Path(sys.argv[2])
    identifier = sys.argv[3]

    scale = None
    if len(sys.argv) > 4:
        scale = float(sys.argv[4])

    run(input_image_path, output_directory, identifier, scale)
