import numpy as np
import typing as tp
import cv2


def get_image_moment(image: np.ndarray, order: int = 1) -> tp.Tuple[float, float]:
    """Compute the image moment of order `order`.

    The image moment is the weighted centre of the image.
    """
    if len(image.shape) > 2:
        image = np.mean(image, axis=2)

    m, n = image.shape
    v = np.mean(image)
    x0 = np.mean(image * np.arange(n)**order) / v
    y0 = np.mean(image * np.arange(m)[:, np.newaxis]**order) / v
    return x0, y0


def grayscale_to_color(color_array: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(color_array, cv2.COLOR_GRAY2BGR)


def color_to_grayscale(color_array: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(color_array, cv2.COLOR_BGR2GRAY)
