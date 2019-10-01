import typing as tp
import pathlib
import dataclasses
import collections
import itertools
import enum

import cv2
import imutils
import numpy as np

import shapely.geometry

class colors(enum.Enum):

    BLUE = (255, 0, 0)
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)

    @property
    def bgr(self):
        return self.value

    def dist(self, other):
        return np.linalg.norm(np.array(self.value) - other)


@dataclasses.dataclass
class Line:
    """Oriented Lines in R^2. """

    a: float
    b: float
    c: float

    def __post_init__(self):
        if self.a == self.b == 0:
            raise ValueError("Parameters 'a' and 'b' cannot both be zero.")

        # Normalize -- Two lines are the same if parameters are the same up to
        # a _positive_ scaling factor, left side is positive

        norm = np.linalg.norm(dataclasses.astuple(self))
        self.a = self.a / norm
        self.b = self.b / norm
        self.c = self.c / norm

    def __call__(self, point) -> float:
        x, y = point
        return self.a * x + self.b * y + self.c

    def __neg__(self) -> "Line":
        return self.__class__(-self.a, -self.b, -self.c)

    def orthogonal_line(self, point) -> "Line":
        """Returns the orthogonal line passing through (x, y). """

        x, y = point
        a, b, c = dataclasses.astuple(self)

        # Orientation consistent with cross-product with pos. z-axis
        return self.__class__(-b, a, -a * y + b * x)

    def project_point(self, point):
        orth = self.orthogonal_line(point)
        return self ^ orth

    def get_line_segment(self, image):
        """ Returns the line segment that intersects the image. """

        # TODO: Fix for when line more closely aligns with y-axis
        m, n = image.shape[:2]
        a, b, c = dataclasses.astuple(self)
        x0, y0 = (0, int(-c/b))
        x1, y1 = (n, int(-(a*n + c)/b))
        return ((x0, y0), (x1, y1))

    def __ge__(self, point):
        return self(point) <= 0

    def __le__(self, point):
        return self(point) >= 0

    def __gt__(self, point):
        return self(point) < 0

    def __lt__(self, point):
        return self(point) > 0

    def __xor__(self, other):
        # Intersect
        A = np.vstack((dataclasses.astuple(self),
                       dataclasses.astuple(other)))

        x = np.linalg.solve(A[:, :2], -A[:, 2])
        return x


@dataclasses.dataclass
class Reader:
    color_to_grayscale: int = cv2.COLOR_BGR2GRAY

    blur_kernel_size: tp.Tuple[int, int] = (3,3)
    blur_dist: int = 0

    morph_kernel = np.ones((3,3))
    morph_num_iter = 1

    thresh_val = 130
    thresh_maxval = 255

    contour_mode: int = cv2.RETR_EXTERNAL
    contour_method: int = cv2.CHAIN_APPROX_TC89_L1

    draw_binary: bool = True
    draw_contours: bool = True
    draw_axis: bool = True

    marker_min_aspect = 0.5
    marker_min_area = 100
    marker_max_value = 100

    def get_image_moment(self, order: int = 1):
        image = self.image
        m, n = image.shape
        v = np.mean(image)
        x0 = np.mean(image * np.arange(n)**order) / v
        y0 = np.mean(image * np.arange(m)[:, np.newaxis]**order) / v
        return (x0, y0)

    def get_axis(self, rectangles: tp.List[np.ndarray]):
        rectangles = [r.reshape(-1, 2) for r in rectangles]

        # First get best line fitting the marker centres
        centres = [r.mean(axis=0) for r in rectangles]
        A = np.ones((len(rectangles), 3))
        for i, c in enumerate(centres):
            A[i, 1:] = c

        # Best line coeeficients from the approximate null-vector
        svd = np.linalg.svd(A)
        (c, a, b) = svd[-1][-1, :]
        centreline = Line(a, b, c)

        # Obtain a better fit using all four vertices of the rectangles
        B = np.zeros((4*len(rectangles), 4))
        for (i, rect) in enumerate(rectangles):
            # Insert rectangle vertices
            B[4 * i: 4*i + 4, 2:] = rect
            for j, pt in enumerate(rect):
                # Constant coefficients
                B[4*i + j, int(centreline <= pt)] = 1

        svd = np.linalg.svd(B)
        (c, d, a, b) = svd[-1][-1, :]

        # Equation for x-axis -- best fit for centreline
        x_axis = Line(a, b, (c+d)/2)

        # Get image gentroid and orient the line
        image_centre = self.get_image_moment(order=1)
        if x_axis >= image_centre:
            x_axis = -x_axis

        # Place a preliminary y-axis
        y_axis_prelim = x_axis.orthogonal_line(image_centre)

        # Place origin on the first marker along the oriented x-axis
        origin = sorted(centres, key=y_axis_prelim)[0]
        y_axis = x_axis.orthogonal_line(origin)

        self.axis = (x_axis, y_axis)

        return x_axis, y_axis

    def match_contours(self,
                       match_types: tp.List[str],
                       contours=None):
        contours = contours or self.get_contours()

        matchers = {type: getattr(self, f"match_{type}") for type in match_types}
        matches = collections.defaultdict(list)

        for c in contours:
            for match_type, matcher in matchers.items():
                match_result = matcher(c)
                if match_result is not None:
                    matches[match_type].append(match_result)
                    break

        return matches

    def match_rectangle(self, c):
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * perimeter, True)

        if len(approx) == 4:

            angles = angles_in_contour(approx)
            # Check angles for rectangle
            if max(abs(angles - np.pi/2)) < 0.1  * np.pi:
                return approx

    def match_marker(self, c):
        rectangle = self.match_rectangle(c)

        if (rectangle is not None
                and rectangle_aspect(rectangle) > self.marker_min_aspect
                and self.get_contour_area(rectangle) > self.marker_min_area
                and self.get_contour_mean_value(rectangle) < self.marker_max_value):
            return rectangle

    def match_graph_candidate(self, c):
        perimeter = cv2.arcLength(c, True)
        pg = shapely.geometry.Polygon(c.reshape(-1, 2))
        rel_area = 4 * pg.area / perimeter**2
        if rel_area < 0.1:
            return c

    def get_contour_area(self, c):
        pg = shapely.geometry.Polygon(c.reshape(-1, 2))
        return pg.area

    def get_contour_interior(self, c):
        c = c.reshape(-1, 2)
        x0, y0 = c.min(axis=0)
        x1, y1 = c.max(axis=0)

        pg = shapely.geometry.Polygon(c)

        pixels = indices_in_window(x0, y0, x1, y1)
        inside = [i for (i, pix) in enumerate(pixels)
                  if pg.contains(shapely.geometry.Point(pix))]

        interior = pixels[inside]
        return interior

    def get_contour_mean_value(self, c):
        I, J = self.get_contour_interior(c).T
        mean_value = self.image_orig[J, I].mean()
        return mean_value

    def get_contour_max_value(self, c):
        I, J = self.get_contour_interior(c).T
        max_value = self.image_orig[J, I].max()
        return max_value

    def load_image(self, filepath: pathlib.Path):
        self.filepath = filepath
        self.image_orig = cv2.imread(str(filepath))
        self.reset_image()
        self.axis = None

    def reset_image(self):
        self.image = self.image_orig.copy()

    def bgr_to_gray(self):
        self.image = cv2.cvtColor(self.image, self.color_to_grayscale)

    def threshold(self, thresh_val=None):
        thresh_val = thresh_val or self.thresh_val
        self.image = cv2.threshold(self.image, self.thresh_val,
                                   self.thresh_maxval, cv2.THRESH_BINARY)[1]

    def blur(self, kernel_size=None):
        kernel_size = kernel_size or self.blur_kernel_size
        self.image = cv2.blur(self.image, self.blur_kernel_size, self.blur_dist)

    def morph(self, kernel=None, iterations=None):
        kernel = kernel or self.morph_kernel
        iterations = iterations or self.morph_num_iter
        self.image = cv2.morphologyEx(self.image, cv2.MORPH_ERODE,
                                      kernel=kernel, iterations=iterations)

    def invert(self):
        self.image = self.thresh_maxval - self.image

    def get_contours(self, min_size: int = 6):
        contours = cv2.findContours(self.image, self.contour_mode, self.contour_method)
        contours = imutils.grab_contours(contours)
        contours = list(filter(lambda c: c.size > min_size, contours))

        return contours

    def draw(self, features):
        image_draw = self.image_orig.copy()
        color_iterator = itertools.cycle(colors)
        lw = 1

        if self.axis:
            color = next(color_iterator)
            x_axis, y_axis = self.axis

            pt0, pt1 = x_axis.get_line_segment(image_draw)
            cv2.line(image_draw, pt0, pt1, color.bgr, lw)

            pt0, pt1 = y_axis.get_line_segment(image_draw)
            cv2.line(image_draw, pt0, pt1, color.bgr, lw)

            x0, y0 = self.get_image_moment()
            cv2.circle(image_draw, (int(x0), int(y0)),  5, color.bgr)
            cv2.circle(image_draw, (int(x0), int(y0)), 25, color.bgr)

        for contours in features.values():
            color = next(color_iterator)
            image_draw = cv2.drawContours(image_draw, contours, -2, color.bgr, lw)

        cv2.imshow("Image", image_draw)
        cv2.waitKey(0)

    def read_image(self, filepath: pathlib.Path):
        self.load_image(filepath)

        # Try to find the black markers in a fairly sharp image
        # First convert to binary grayscale image and convert
        self.bgr_to_gray()
        self.threshold()
        self.invert()

        features = self.match_contours(match_types=["marker"])

        # Now we can find axes
        markers = features["marker"]
        self.get_axis(markers)

        # Reset and find graphs
        self.reset_image()
        self.bgr_to_gray()
        self.threshold()
        self.morph()
        self.invert()

        features.update(self.match_contours(match_types=["graph_candidate"]))

        self.draw(features)

        return features


def indices_in_window(x0, y0, x1, y1):
    I, J = np.meshgrid(np.arange(x0, x1+1), np.arange(y0, y1+1))
    return np.hstack((I.reshape(-1, 1), J.reshape(-1,1)))


def angles_in_contour(c):
    """Get angles of a convex contour. """
    pg = c.reshape(-1, 2)
    # Get normalized edge vectors
    n = pg - np.roll(pg, -1, axis=0)
    n = n / np.linalg.norm(n, axis=1).reshape(-1, 1)

    # Compute cosine of exterior angles
    # NOTE: Would need to handle sign for reentrant corners (use atan2)
    r = np.clip((n * np.roll(n, -1, axis=0)).sum(axis=1), -1, 1)

    # Get interior angles
    angles = np.pi - np.arccos(r)
    return angles


def rectangle_aspect(c):
    pg = c.reshape(-1, 2)
    dx = np.linalg.norm(pg[1] - pg[0])
    dy = np.linalg.norm(pg[3] - pg[0])
    return min(dx, dy) / max(dx, dy)

if __name__ == "__main__":
    reader = Reader()

    filepath = pathlib.Path("data/scan1.png")
    conts = reader.read_image(filepath)

