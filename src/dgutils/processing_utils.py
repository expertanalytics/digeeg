import numpy as np
import typing as tp

import cv2
import time
import logging

from pathlib import Path

from dgimage import Image

from .utils import (
    get_contours,
    match_contours,
    save,
)

from .matchers import get_marker_matcher

from .debug import get_debug_path


log = logging.getLogger(__name__)


def remove_structured_background(
    *,
    image_array: np.ndarray,
    background_kernel_size: tp.Tuple[int, int],
    smoothing_kernel_size: tp.Tuple[int, int],
    debug: bool = True
) -> None:
    """Remove structured background features conforming to `background_kernel_size`.

    NB! The foreground should be white
    """
    if debug:
        debug_path = get_debug_path("markers")
        save(image_array, debug_path, "input")

    structuring_element = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        background_kernel_size
    )

    cv2.erode(image_array, structuring_element, dst=image_array)
    cv2.dilate(image_array, structuring_element, dst=image_array)

    # 1. Extract edges
    edges = cv2.adaptiveThreshold(
        image_array,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        3,
        -2
    )

    # 2. Dilate edges
    kernel = np.ones(smoothing_kernel_size)
    edges = cv2.dilate(edges, kernel)

    # 4. blur smooth img
    smooth = cv2.blur(image_array, smoothing_kernel_size)

    # 5 smooth.copyTo(src, edges)
    # src, mask, dst -> dst
    image_array[:] = cv2.copyTo(smooth, image_array)       # I think this is right
    if debug:
        debug_path = get_debug_path("remove_structured_background")
        save(image_array, debug_path, "output")


def markers(
    image: Image,
    *,
    kernel_length: int = 0,
    blur_kernel_size: int = 9,
    threshold_value: float = 175,
    num_iterations: int = 4,
    debug: bool = False
) -> tp.List[np.ndarray]:
    """Return the contours of the black square markers."""

    from .plots import plot
    # cv2.imwrite("foo.png", image.image)
    # assert False, image.image.dtype

    if debug:
        debug_path = get_debug_path("markers")
        save(image.image, debug_path, "input")

    assert len(image.image.shape) == 2, f"Expecting binary image"
    image.invert()
    image.blur(blur_kernel_size)
    image.threshold(threshold_value)

    if debug:
        save(image.image, debug_path, "threshold")

    if kernel_length > 0 and num_iterations > 0:
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
        horisontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))

        # vertial lines
        vertical_image = cv2.erode(image.image, vertical_kernel, iterations=num_iterations)
        cv2.dilate(vertical_image, vertical_kernel, iterations=num_iterations, dst=vertical_image)

        # Horisontal lines
        horisontal_image = cv2.erode(image.image, horisontal_kernel, iterations=num_iterations)
        cv2.dilate(image.image, horisontal_kernel, iterations=num_iterations, dst=vertical_image)

        # Compute intersection of horisontal and vertical
        cv2.bitwise_and(horisontal_image, vertical_image, dst=image.image)

    contours = get_contours(image=image, min_size=4)
    if  len(contours) == 0:
        cv2.imwrite("foo.png", image.image)
    if debug:
        image_draw = Image(image.copy_image())
        image_draw.gray_to_bgr()
        image_draw = cv2.drawContours(image_draw.image, contours, -2, (0, 255, 0), 2)
        save(image_draw, debug_path, "morphed")

    features = match_contours(matcher=get_marker_matcher(image=image), contours=contours)
    if debug:
        image_draw = Image(image.copy_image())
        image_draw.gray_to_bgr()
        image_draw = cv2.drawContours(image_draw.image, features, -2, (0, 0, 255), 2)
        save(image.image, debug_path, "features")
    image.reset_image()
    return features
