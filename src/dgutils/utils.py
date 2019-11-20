import numpy as np
import cv2
import shapely
import typing as tp


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


def match_square(contour, tol=0.2):
    """Return True if contour is approximately bounded by a square."""
    x, y, w, h = cv2.boundingRect(contour)
    if 1 - tol < w / h < 1 + tol:
        return True
    return False


def contour_interior(contour: np.ndarray):
    contour = contour.reshape(-1, 2)
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
) -> tp.List[tp.Any]:
    matches = []

    for c in contours:
        match_result = matcher(c)
        if match_result is not None:
            matches.append(match_result)

    if len(matches) > 0:
        return matches


def filter_contours(image: "Image", contours: tp.Sequence[np.ndarray]):
    """Keep only the pixels inside the contours."""
    if len(contours) == 0:
        assert False, "No contours"

    shape = (m, n) = image.image.shape[:2]
    image_filter = np.zeros(shape, dtype=image.image.dtype)
    image_filter = cv2.drawContours(image_filter, contours, -2, 255, cv2.FILLED)

    _, binary_image = cv2.threshold(image.image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.bitwise_and(image_filter, binary_image, dst=image.image)


def remove_contours(image: "Image", contours: tp.Sequence[np.ndarray], fill_value: int = None):
    """Remove the pixels inside the contours."""
    if len(contours) == 0:
        assert False, "No contours"
    shape = m, n, *_ = image.image.shape

    _fill_value = fill_value
    if _fill_value is None:
        _fill_value = np.argmax(np.bincount(image.image.ravel()))

    image.image = cv2.drawContours(image.image, contours, -2, int(_fill_value), cv2.FILLED)


def filter_image(image: "Image", binary_mask: np.ndarray, fill_value: int = None):
    _fill_value = fill_value

    if binary_mask.dtype != bool or not np.in1d(np.unique(binary_mask), (0, 1)).all():
        assert False, "Binary mask must be boolean or contain only {0 1}."""

    if _fill_value is None:
        _fill_value = np.argmax(np.bincount(image.image.ravel()))

    image.image[binary_mask] = _fill_value


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


def get_bounding_rectangle_matcher():

    def matcher(contour):
        x, y, w, h = list(map(int, cv2.boundingRect(contour)))
        aspect_ratio = float(w)/h

        area = cv2.contourArea(contour)
        hull = cv2.convexHull(contour)

        hull_area = cv2.contourArea(hull)
        solidity = float(area)/hull_area

        (x, y),(MA, ma), angle = cv2.fitEllipse(contour)

        slash = 20 < angle < 40
        horisontal = 80 < angle < 100 or aspect_ratio > 1
        solid = solidity > 0.5

        if solid and (horisontal or slash):
            rectangle = cv2.minAreaRect(contour)
            rectangle_contour = np.int0(cv2.boxPoints(rectangle))
            return rectangle_contour
    return matcher
