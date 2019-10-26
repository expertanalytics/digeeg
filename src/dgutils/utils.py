import numpy as np
import shapely


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
