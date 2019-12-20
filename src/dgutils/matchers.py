import numpy as np
import typing as tp

import cv2
import shapely.geometry
import logging

from .utils import (
    match_rectangle,
    rectangle_aspect_ratio,
    get_contour_mean_value,
    angles_in_contour,
)

from dgimage import Image


log = logging.getLogger(__name__)


def get_marker_matcher(
    *,
    image: Image,
    marker_min_aspect: float = 0.5,
    marker_min_area: float = 100,
    marker_max_value: float = 100
) -> tp.Callable[[np.ndarray], np.ndarray]:

    def matcher(contour: np.ndarray) -> np.ndarray:
        """Return rectange if it matches a square marker on the paper strip."""
        rectangle = match_rectangle(contour)

        if (rectangle is not None
                and rectangle_aspect_ratio(rectangle) > marker_min_aspect
                and cv2.contourArea(rectangle) > marker_min_area
                and get_contour_mean_value(image, rectangle) < marker_max_value):
            return rectangle

    return matcher


def get_square_matcher(
    *,
    approx_tolerance: float = 0.04,
    angle_tolerance: float = 0.1
) -> tp.Callable[[np.ndarray], np.ndarray]:
    """Angle tolerance is in fractions of pi."""

    def matcher(contour: np.ndarray):
        perimeter = cv2.arcLength(contour, True)

        # create a closed polygonal approximation to `c` with the distance between them
        # less than `epsilon*perimeter`.
        approx = cv2.approxPolyDP(contour, approx_tolerance*perimeter, True)
        if len(approx) == 4:
            angles = angles_in_contour(approx)
            if max(abs(angles - np.pi/2)) < angle_tolerance*np.pi:
                return approx

    return matcher


def get_graph_matcher(
        *,
        approximation_tolerance: float = 0.05
) -> tp.Callable[[np.ndarray], np.ndarray]:

    def matcher(contour: np.ndarray) -> np.ndarray:
        """Return contour if it is poorly approximated by a circle."""
        perimeter = cv2.arcLength(contour, True)
        pg = shapely.geometry.Polygon(contour.reshape(-1, 2))
        rel_area = 4 * pg.area / perimeter**2
        if rel_area < approximation_tolerance:
            return contour

    return matcher


def get_bounding_rectangle_matcher(
    *,
    min_solidity: float = 0.5,
    min_aspect_ratio: float = 1.0,
    min_shlash_angle: int = 20,
    max_shlash_angle: int = 40,
    min_horisontal_elipse_angle: int = 80,
    max_horisontal_elipse_angle: int = 80
) -> tp.Callable[[np.ndarray], np.ndarray]:

    def matcher(contour):
        x, y, w, h = list(map(int, cv2.boundingRect(contour)))
        aspect_ratio = float(w)/h

        area = cv2.contourArea(contour)
        hull = cv2.convexHull(contour)

        hull_area = cv2.contourArea(hull)
        solidity = float(area)/hull_area

        (x, y),(MA, ma), angle = cv2.fitEllipse(contour)

        slash = min_shlash_angle < angle < max_shlash_angle
        horisontal = (min_horisontal_elipse_angle < angle < max_horisontal_elipse_angle
                      or aspect_ratio > min_aspect_ratio)
        solid = solidity > min_solidity

        if solid and (horisontal or slash):
            rectangle = cv2.minAreaRect(contour)
            rectangle_contour = np.int0(cv2.boxPoints(rectangle))
            return rectangle_contour

    return matcher


