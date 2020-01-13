import numpy as np
import typing as tp

import cv2
import time
import shapely
import imutils
import logging


from pathlib import Path

from dgimage import Image


log = logging.getLogger(__name__)


def angles_in_contour(contour: np.ndarray) -> np.ndarray:
    """Get angles of a convex contour."""
    pg = contour.reshape(-1, 2)

    # Get normalized edge vectors
    n = pg - np.roll(pg, -1, axis=0)
    n = n / np.linalg.norm(n, axis=1).reshape(-1, 1)

    # Compute cosine of exterior angles
    # NOTE: Would need to handle sign for reentrant corners (use atan2)
    r = np.clip((n * np.roll(n, -1, axis=0)).sum(axis=1), -1, 1)

    # Get interior angles
    angles = np.pi - np.arccos(r)
    return angles


def indices_in_window(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    I, J = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
    return np.hstack((I.reshape(-1, 1), J.reshape(-1, 1)))


def rectangle_aspect_ratio(contour: np.ndarray) -> float:
    """Return the aspect ratio of a rectangle."""
    pg = contour.reshape(-1, 2)
    dx = np.linalg.norm(pg[1] - pg[0])
    dy = np.linalg.norm(pg[3] - pg[0])
    return min(dx, dy) / max(dx, dy)


def match_square(contour, aspect_ratio_tolerance: float = 0.2) -> np.ndarray:
    """Return True if contour is approximately bounded by a square."""
    x, y, w, h = cv2.boundingRect(contour)
    if 1 - aspect_ratio_tolerance < w / h < 1 + aspect_ratio_tolerance:
        return True
    return False


def contour_interior(contour: np.ndarray) -> np.ndarray:
    contour = contour.reshape(-1, 2)    # copy the contour
    x0, y0 = pg.min(axis=0)
    x1, y1 = pg.max(axis=0)

    pg = shapely.geometry.Polygon(contour)
    pixels = indices_in_window(x0, y0, x1, y1)
    inside = [
        i for (i, pix) in enumerate(pixels) if pg.contains(shapely.geometry.Point(pix))
    ]

    interior = pixels[inside]
    return interior


def match_contours(
    *,
    matcher: tp.Callable,
    contours: tp.Sequence[np.ndarray]
) -> tp.List[np.ndarray]:
    """Filter the conturs by the supplied matching function."""
    matches = []

    for c in contours:
        match_result = matcher(c)
        if match_result is not None:
            matches.append(match_result)

    if len(matches) > 0:
        return matches


def filter_contours(
    *,
    image_array: np.ndarray,
    contours: tp.Sequence[np.ndarray]
) -> None:
    """Keep only the pixels inside the contours.


    `cv2.copyTo` assumes that the background is black, so set `invert` appropriately.
    """
    if len(contours) == 0:
        assert False, "No contours"

    shape = (m, n) = image_array.shape[:2]
    image_filter = np.zeros(shape, dtype=image_array.dtype)
    image_filter = cv2.drawContours(image_filter, contours, -2, 255, cv2.FILLED)

    # image.image = cv2.copyTo(image, mask=image_filter)
    foo = np.sum(image_array)
    image_array[:] = cv2.copyTo(image_array, mask=image_filter)
    assert np.sum(image_array) != foo

    # _, binary_image = cv2.threshold(image.image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2.bitwise_and(image_filter, binary_image, dst=image.image)


def remove_contours(image: Image, contours: tp.Sequence[np.ndarray], fill_value: int = None):
    """Remove the pixels inside the contours."""
    if len(contours) == 0:
        assert False, "No contours"
    shape = m, n, *_ = image.image.shape

    _fill_value = fill_value
    if _fill_value is None:
        _fill_value = np.argmax(np.bincount(image.image.ravel()))

    image.image = cv2.drawContours(image.image, contours, -2, int(_fill_value), cv2.FILLED)


def filter_image(
    *,
    image_array: np.ndarray,
    binary_mask: np.ndarray,
    fill_value: int = None
) -> None:
    _fill_value = fill_value

    if binary_mask.dtype != bool:
        assert False, "Binary mask must be boolean."""

    if _fill_value is None:
        _fill_value = np.argmax(np.bincount(image_array.ravel()))

    image_array[binary_mask] = _fill_value


def get_contour_interior(contour: np.ndarray) -> np.ndarray:
    _c = contour.reshape(-1, 2)
    x0, y0 = _c.min(axis=0)
    x1, y1 = _c.max(axis=0)

    pg = shapely.geometry.Polygon(_c)

    pixels = indices_in_window(x0, y0, x1, y1)
    inside = [i for (i, pix) in enumerate(pixels)
              if pg.contains(shapely.geometry.Point(pix))]

    interior = pixels[inside]
    return interior


def get_contour_max_value(image: Image, contour: np.ndarray) -> float:
    I, J = get_contour_interior(contour).T
    max_value = image.image_orig[J, I].max()
    return max_value


def get_contour_mean_value(self, c: np.ndarray) -> float:
    I, J = get_contour_interior(c).T
    mean_value = self.image_orig[J, I].mean()
    return mean_value


def image_to_point_cloud(image: Image) -> np.ndarray:
    """Return a scatterpolot of the image."""
    I, x = np.where(image.image)
    y = image.image.shape[0] - I - 1
    return np.hstack((x.reshape(-1,1), y.reshape(-1,1)))


def get_contours(
    *,
    image: Image,
    min_size: int = 6,
    contour_mode: int = cv2.RETR_EXTERNAL,       # Retreive only the external contours
    contour_method: int = cv2.CHAIN_APPROX_TC89_L1      # Apply a flavor of the Teh Chin chain approx algo
) -> tp.List[np.ndarray]:
    """Find contours in a binary image."""
    contours = cv2.findContours(image.image, contour_mode, contour_method)
    contours = imutils.grab_contours(contours)
    contours = list(filter(lambda c: c.shape[0] > min_size, contours))
    return contours


def match_rectangle(c: np.ndarray, rectangle_approx_tol: float = 0.04):
    """If `c` can be approximated by a square, return an approximation."""
    perimeter = cv2.arcLength(c, True)

    # create a closed polygonal approximation to `c` with the distance between them
    # less than `epsilon*perimeter`.
    approx = cv2.approxPolyDP(c, rectangle_approx_tol*perimeter, True)

    if len(approx) == 4:
        angles = angles_in_contour(approx)
        # Check angles for rectangle
        if max(abs(angles - np.pi/2)) < 0.1  * np.pi:       # 0.1
            return approx


def save(image_array: np.ndarray, directory: Path, name: str) -> None:
    """Save `image_array` in `path` with name as a png."""
    directory.mkdir(exist_ok=True, parents=True)

    # TODO: Problems with png resolution
    cv2.imwrite(str(directory / f"{name}.png"), image_array)
