import numpy as np

import cv2
import shapely.geometry

from .utils import (
    match_rectangle,
    rectangle_aspect_ratio,
    get_contour_mean_value,
    angles_in_contour,
)


def get_marker_matcher(
    image: "Image",
    marker_min_aspect: float = 0.5,
    marker_min_area: float = 100,
    marker_max_value: float = 100
):

    def matcher(contour: np.ndarray) -> np.ndarray:
        """Return rectange if it matches a square marker on the paper strip."""
        rectangle = match_rectangle(contour)

        if (rectangle is not None
                and rectangle_aspect_ratio(rectangle) > marker_min_aspect
                and cv2.contourArea(rectangle) > marker_min_area
                and get_contour_mean_value(image, rectangle) < marker_max_value):
            return rectangle
    return matcher


def get_square_matcher(*, approx_tolerance: float = 0.04, angle_tolerance: float = 0.1):
    """Angle tolerance is in fractions of pi."""

    def matcher(contour: np.ndarray):
        perimeter = cv2.arcLength(contour, True)

        # create a closed polygonal approximation to `c` with the distance between them
        # less than `epsilon*perimeter`.
        approx = cv2.approxPolyDP(contour, approx_tolerance*perimeter, True)
        if len(approx) == 4:
            angles = angles_in_contour(approx)
            if max(abs(angles - np.pi/2)) < angle_tolerance*np.pi:       # 0.1
                return approx
    return matcher


def get_graph_matcher(*, approximation_tolerance: float = 0.05):

    def matcher(contour: np.ndarray) -> np.ndarray:
        """Return contour if it is poorly approximated by a circle."""
        perimeter = cv2.arcLength(contour, True)
        pg = shapely.geometry.Polygon(contour.reshape(-1, 2))
        rel_area = 4 * pg.area / perimeter**2
        if rel_area < approximation_tolerance:
            return contour

    return matcher
