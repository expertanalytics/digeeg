import numpy as np
import typing as tp

import cv2


def remove_structured_background(
    *,
    image_array: np.ndarray,
    background_kernel_size: tp.Tuple[int, int],
    smoothing_kernel_size: tp.Tuple[int, int]
) -> None:
    """Remove structured background features conforming to `background_kernel_size`.

    NB! The foreground should be white
    """

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
