import numpy as np
import typing as tp

import dataclasses
import cv2
import shapely

from collections import defaultdict

# local stuff
from dgimage import Image

from dgutils import (
    angles_in_contour,
    rectangle_aspect_ratio,
    contour_interior,
    indices_in_window,
)

from dgresample import Line


def match_rectangle(contour: np.ndarray, approximation_tolerance: float = 0.04):
    """If `c` can be approximated by a square, return an approximation."""
    perimeter = cv2.arcLength(contour, True)

    # create a closed polygonal approximation to `c` with the distance between them
    # less than `epsilon*perimeter`.
    approx = cv2.approxPolyDP(contour, approximation_tolerance*perimeter, True)

    if len(approx) == 4:
        angles = angles_in_contour(approx)
        # Check angles for rectangle
        if max(abs(angles - np.pi/2)) < 0.1  * np.pi:
            return approx


def match_marker(
    image: Image,
    contour: np.ndarray,
    min_aspect_ratio: float,    # 0.5
    min_area: float,            # 100
    max_value: float            # 100
) -> np.ndarray:
    """Return rectange if it matches a square marker on the paper strip."""
    rectangle = match_rectangle(contour)

    if (
        rectangle is not None and
        rectangle_aspect_ratio(rectangle) > min_aspect_ratio and
        cv2.contourArea(rectangle) > min_area and
        image.reduce_contour(rectangle, operation=np.mean) < max_value
    ):
        return rectangle


def filter_match_contours(
    *,
    contours: tp.Iterable[np.ndarray],
    matcher: tp.Callable[[np.ndarray], tp.Any]
) -> tp.List[tp.Any]:
    """Itarate over `contours` and apply `matcher` for each one.

    Return list of non None results.
    """
    match_list = []
    for c in contours:
        match_result = matcher(c)
        if match_result is not None:
            match_list.append(match_result)
    return match_list


def get_axis(image_centre: tp.Tuple[float, float], rectangles: tp.List[np.ndarray]):
    rectangles = [r.reshape(-1, 2) for r in rectangles]

    # First get best line fitting the marker centres
    centres = [r.mean(axis=0) for r in rectangles]
    A = np.ones((len(rectangles), 3))
    for i, c in enumerate(centres):
        A[i, 1:] = c

    # Best line coeeficients from the approximate null-vector
    svd = np.linalg.svd(A)
    c, a, b = svd[-1][-1, :]
    centreline = Line(a, b, c)

    # Obtain a better fit using all four vertices of the rectangles
    B = np.zeros((4*len(rectangles), 4))
    for i, rect in enumerate(rectangles):
        # Insert rectangle vertices
        B[4 * i: 4*i + 4, 2:] = rect
        for j, pt in enumerate(rect):
            # Constant coefficients
            B[4*i + j, int(centreline <= pt)] = 1

    svd = np.linalg.svd(B)
    c, d, a, b = svd[-1][-1, :]

    # Equation for x-axis -- best fit for centreline
    # x_axis = Line(a, b, (c + d)/2)
    x_axis = Line(a, b, c)      # Why not (c + d) / 2

    # Get image gentroid and orient the line
    if x_axis >= image_centre:
        x_axis = -x_axis

    # Place a preliminary y-axis
    y_axis_prelim = x_axis.orthogonal_line(image_centre)

    # Place origin on the first marker along the oriented x-axis
    origin = sorted(centres, key=y_axis_prelim)[0]
    y_axis = x_axis.orthogonal_line(origin)

    axis = x_axis, y_axis
    scale = np.linalg.norm(centres[1] - centres[0])
    return axis, scale


def resample(axis):
    x_axis, y_axis = axis

    origin = x_axis ^ y_axis

    a = x_axis ^ (y_axis - self.scale * self.resample_x_max)
    b = (x_axis - self.scale * self.resample_y_max) ^ y_axis

    n_x = int(self.resample_x_max / self.resample_step_x)
    n_y = int(self.resample_y_max / self.resample_step_y)

    # Get the coordinates defining affine transform
    source_pts = np.array([origin, a, b], dtype="float32")
    target_pts = np.array([[0,n_y], [n_x, n_y], [0, 0]], dtype="float32")
    mapping = cv2.getAffineTransform(source_pts, target_pts)

    self.image = cv2.warpAffine(self.image, mapping, (n_x, n_y))
