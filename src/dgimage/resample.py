import numpy as np
import typing as tp

import dataclasses
import cv2
import logging

from .line import Line
from .image import Image
from .image_utils import get_image_moment


logger = logging.getLogger(__name__)


def get_axis(image: Image, rectangles: tp.List[np.ndarray]):
    if len(rectangles) < 2:
        assert False, "Cannot scale axis based on only one marker."
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
    for (i, rect) in enumerate(rectangles):
        # Insert rectangle vertices
        B[4 * i: 4*i + 4, 2:] = rect
        for j, pt in enumerate(rect):
            # Constant coefficients
            # Need two constatn coefficients because there is a vertical distance
            # between the top and bottom corners
            B[4*i + j, int(centreline <= pt)] = 1

    svd = np.linalg.svd(B)
    c, d, a, b = svd[-1][-1, :]

    # Equation for x-axis -- best fit for centreline
    x_axis = Line(a, b, (c + d)/2)
    # x_axis = Line(a, b, max(c, d))      # use min(c, d) to get closes to edge of image

    # Get image gentroid and orient the line
    image_centre = get_image_moment(image.image, order=1)
    if x_axis >= image_centre:
        x_axis = -x_axis

    # Place a preliminary y-axis at the image centre
    y_axis_prelim = x_axis.orthogonal_line(image_centre)

    w, h, *_ = image.image.shape
    if w / h > 1:       # Figure out the orientation of the image
        # Where does the x-axis intersect the image border?
        intersection = -x_axis.b*w/x_axis.a - x_axis.c/x_axis.a
        # Set y-axis at the correct image border
        if y_axis_prelim >= (intersection, w):
            point = (intersection, w)
        else:
            point = (intersection, 0)
    else:       # w / h < 1
        intersection = -x_axis.c/x_axis.b
        if y_axis_prelim >= (0, intersection):
            point = (0, intersection)
        else:
            point = (h, intersection)

    y_axis = x_axis.orthogonal_line(point)
    axis = (x_axis, y_axis)
    scale = np.linalg.norm(centres[1] - centres[0])
    return axis, scale


def resample(
    image: Image,
    *,
    x_max: float = 2.0,
    y_max: float = 0.8,
    step_x: float = 1e-3,
    step_y: float  = 1e-3
) -> None:
    # TODO: To some error checking here
    x_axis, y_axis = image.axis
    scale = image.scale
    origin = x_axis ^ y_axis        # Intersection

    # subtrcting from 'c' moves along the line
    a = x_axis ^ (y_axis - scale * x_max)
    b = (x_axis - image.scale * y_max) ^ y_axis

    n_x = int(x_max/step_x)
    n_y = int(y_max/step_y)

    # Get the coordinates defining affine transform
    source_pts = np.array([origin, a, b], dtype="float32")
    target_pts = np.array([[0,n_y], [n_x, n_y], [0, 0]], dtype="float32")
    mapping = cv2.getAffineTransform(source_pts, target_pts)

    image.invert()
    image.image = cv2.warpAffine(image.image, mapping, (n_x, n_y))
    image.invert()
