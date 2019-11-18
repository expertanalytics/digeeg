import numpy as np
import typing as tp

import dataclasses
import cv2
import shapely

from collections import defaultdict

from dgutils import (
    angles_in_contour,
    rectangle_aspect_ratio,
    contour_interior,
    indices_in_window,
)

from dgutils import Line


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
    contour: np.ndarray,
    min_aspect_ratio: float,    # 0.5
    min_area: float,            # 100
    max_value: float            # 100
) -> np.ndarray:
    """Return rectange if it matches a square marker on the paper strip."""
    rectangle = match_rectangle(contour)

    # if rectangle is not None:
    #     foo = rectangle_aspect_ratio(rectangle) > min_aspect_ratio
    #     bar = cv2.contourArea(rectangle)
    #     foo > min_aspect_ratio, bar > min_area
    #     # baz = image.reduce_contour(rectangle, operation=np.mean)
    #     # print(baz)

    if (
        rectangle is not None and
        rectangle_aspect_ratio(rectangle) > min_aspect_ratio and
        cv2.contourArea(rectangle) > min_area
        # and image.reduce_contour(rectangle, operation=np.mean) < max_value
    ):
        return rectangle


def match_graph_candidate(contour: np.ndarray) -> np.ndarray:
    """Return contour if it is poorly approximated by a circle."""
    perimeter = cv2.arcLength(contour, True)
    pg = shapely.geometry.Polygon(contour.reshape(-1, 2))
    rel_area = 4*pg.area/perimeter**2
    if rel_area < 0.1:
        return contour


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

