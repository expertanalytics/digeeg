import typing as tp
from pathlib import Path
import dataclasses
import collections
import itertools

import operator

from dgutils import (
    rectangle_aspect_ratio,
    angles_in_contour,
)

import cv2
import imutils
import numpy as np

import shapely.geometry

import matplotlib.pyplot as plt


from dgutils import Line

from .colors import colors


@dataclasses.dataclass
class Image:
    image_orig: np.ndarray
    image: np.ndarray = None

    grayscale_to_color: int = cv2.COLOR_GRAY2BGR

    blur_kernel_size: int = (3, 3)
    blur_dist: int = 0

    thresh_val = 130
    thresh_maxval = 255

    contour_mode: int = cv2.RETR_EXTERNAL       # Retreive only the external contours
    contour_method: int = cv2.CHAIN_APPROX_TC89_L1      # Apply a flavor of the Teh Chin chain approx algo

    draw_binary: bool = True
    draw_contours: bool = True
    draw_axis: bool = True

    marker_min_aspect = 0.5
    marker_min_area = 100
    marker_max_value = 100

    rectangle_approx_tol = 0.04

    resample_x_max = 2.0
    resample_y_max = 0.8
    resample_step_x = 1/1000
    resample_step_y = 1/1000

    axis = None
    scale = None

    def __post_init__(self):
        self.checkpoint_dict: tp.Dict[str, np.ndarray] = dict()
        self.reset_image()

    def checkpoint(self, tag: str):
        """Set image_orig to current image."""
        self.checkpoint_dict[tag] = self.copy_image()

    def copy_image(self):
        """Return a copy of `self.image`."""
        return self.image.copy()

    def get_image_moment(self, order: int = 1):
        image = self.image
        if len(image.shape) > 2:
            image = np.mean(image, axis=2)

        m, n = image.shape
        v = np.mean(image)
        x0 = np.mean(image * np.arange(n)**order) / v
        y0 = np.mean(image * np.arange(m)[:, np.newaxis]**order) / v
        return (x0, y0)

    def get_axis(self, rectangles: tp.List[np.ndarray]):
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
        # x_axis = Line(a, b, (c + d)/2)
        x_axis = Line(a, b, min(c, d))      # use min(c, d) to get closes to edge of image

        # Get image gentroid and orient the line
        image_centre = self.get_image_moment(order=1)
        if x_axis >= image_centre:
            x_axis = -x_axis

        # Place a preliminary y-axis at the image centre
        y_axis_prelim = x_axis.orthogonal_line(image_centre)

        w, h, *_ = self.image.shape
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
        self.axis = (x_axis, y_axis)
        self.scale = np.linalg.norm(centres[1] - centres[0])

    def match_rectangle(self, c: np.ndarray):
        """If `c` can be approximated by a square, return an approximation."""
        perimeter = cv2.arcLength(c, True)

        # create a closed polygonal approximation to `c` with the distance between them
        # less than `epsilon*perimeter`.
        approx = cv2.approxPolyDP(c, self.rectangle_approx_tol*perimeter, True)

        if len(approx) == 4:
            angles = angles_in_contour(approx)
            # Check angles for rectangle
            if max(abs(angles - np.pi/2)) < 0.1  * np.pi:       # 0.1
                return approx

    def match_marker(self, c: np.ndarray) -> np.ndarray:
        """Return rectange if it matches a square marker on the paper strip."""
        rectangle = self.match_rectangle(c)

        if (rectangle is not None
                and rectangle_aspect_ratio(rectangle) > self.marker_min_aspect
                and self.get_contour_area(rectangle) > self.marker_min_area
                and self.get_contour_mean_value(rectangle) < self.marker_max_value):
            return rectangle

    def match_graph_candidate(self, c: np.ndarray) -> np.ndarray:
        """Return contour if it is poorly approximated by a circle."""
        perimeter = cv2.arcLength(c, True)
        pg = shapely.geometry.Polygon(c.reshape(-1, 2))
        rel_area = 4 * pg.area / perimeter**2
        if rel_area < 0.05:
            return c

    def get_contour_area(self, c):
        return cv2.contourArea(c)

    def get_contour_interior(self, c):
        _c = c.reshape(-1, 2)
        x0, y0 = _c.min(axis=0)
        x1, y1 = _c.max(axis=0)

        pg = shapely.geometry.Polygon(_c)

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

    def reset_image(self, tag: str = None) -> None:

        self.image = self.checkpoint_dict.get(tag, self.image_orig).copy()
        # self.image = self.image_orig.copy()

    def bgr_to_gray(self) -> None:
        """Convert image to greyscale."""
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def gray_to_bgr(self):
        self.image = cv2.cvtColor(self.image, self.grayscale_to_color)

    def threshold(self, thresh_val: float = None) -> None:
        """Apply a fixed level threshold to each pixel.

        dst(x, y) = maxval if src(x, y) > thresh_val else 0

        thresh_val is set from the image histogram using Otsu's binarisation, assuming the image
        histogram is bimodal.

        It is recommended to blur the image before binarisation.
        """
        if thresh_val is None:
            _, self.image = cv2.threshold(self.image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, self.image = cv2.threshold(self.image, thresh_val, 255, cv2.THRESH_BINARY)

    def gaussian_blur(self, kernel_size: int):
        cv2.GaussianBlur(self.image, (kernel_size, kernel_size), 0, dst=self.image)

    def blur(self, kernel_size: int):
        cv2.blur(self.image, (kernel_size, kernel_size), dst=self.image)

    def morph(self, transform: int, kernel_size: tp.Tuple[int, int], iterations: int):
        """
        Valid transforms include:
         - cv2.MORPH_ERODE
         - cv2.MORPH_OPEN
         - cv2.MORPH_CLOSE
         - cv2.MORPH_DILATE

        """
        kernel = np.ones(kernel_size)
        self.image = cv2.morphologyEx(
            self.image,
            transform,
            kernel=kernel,
            iterations=iterations
        )

    def invert(self) -> None:
        """Invert a binary greyscale image."""
        self.image = self.thresh_maxval - self.image

    def get_contours(self, min_size: int = 6) -> tp.List[np.ndarray]:
        """Find contours in a binary image."""
        contours = cv2.findContours(self.image, self.contour_mode, self.contour_method)
        contours = imutils.grab_contours(contours)
        contours = list(filter(lambda c: c.size > min_size, contours))

        return contours

    def resample(self):
        x_axis, y_axis = self.axis
        origin = x_axis ^ y_axis        # Intersection

        # subtrcting from 'c' moves along the line
        a = x_axis ^ (y_axis - self.scale * self.resample_x_max)
        b = (x_axis - self.scale * self.resample_y_max) ^ y_axis

        n_x = int(self.resample_x_max / self.resample_step_x)
        n_y = int(self.resample_y_max / self.resample_step_y)

        # Get the coordinates defining affine transform
        source_pts = np.array([origin, a, b], dtype="float32")
        target_pts = np.array([[0,n_y], [n_x, n_y], [0, 0]], dtype="float32")
        mapping = cv2.getAffineTransform(source_pts, target_pts)

        self.invert()
        self.image = cv2.warpAffine(self.image, mapping, (n_x, n_y))
        self.invert()

    def image_to_point_cloud(self):
        I, x = np.where(self.image)
        y = self.image.shape[0] - I - 1
        return np.hstack((x.reshape(-1,1), y.reshape(-1,1)))

    def draw(self, features, draw_axis=True, show=True):
        color_iterator = itertools.cycle(colors)
        lw = 1

        image_draw = self.copy_image()
        if len(image_draw.shape) < 3:
            image_draw = cv2.cvtColor(image_draw, self.grayscale_to_color)

        if self.axis and not draw_axis:
            color = next(color_iterator)
            x_axis, y_axis = self.axis

            pt0, pt1 = x_axis.get_line_segment(image_draw)
            cv2.line(image_draw, pt0, pt1, color.bgr, lw)

            pt0, pt1 = y_axis.get_line_segment(image_draw)
            cv2.line(image_draw, pt0, pt1, color.bgr, lw)

            pt0, pt1 = (y_axis -self.scale * self.resample_x_max).get_line_segment(image_draw)
            cv2.line(image_draw, pt0, pt1, color.bgr, lw)

            pt0, pt1 = (x_axis -self.scale * self.resample_y_max).get_line_segment(image_draw)
            cv2.line(image_draw, pt0, pt1, color.bgr, lw)

            x0, y0 = self.get_image_moment()
            cv2.circle(image_draw, (int(x0), int(y0)),  5, color.bgr)
            cv2.circle(image_draw, (int(x0), int(y0)), 25, color.bgr)

        color = next(color_iterator)
        image_draw = cv2.drawContours(image_draw, features, -2, color.bgr, lw)

        if show:
            cv2.imshow("Image", image_draw)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return image_draw

    def save_image(self, filepath: Path):
        success = cv2.imwrite(str(filepath), self.image)
        if not success:
            print("Failed to save image")

    def show(self, image=None, mode=None):
        if mode == cv2.WINDOW_NORMAL:
            cv2.namedWindow("ImageShow", cv2.WINDOW_NORMAL)

        if image is None:
            _image_show = self.image
        else:
            _image_show = image

        # cv2.namedWindow("ImageShow", cv2.WINDOW_NORMAL)
        cv2.imshow("ImageShow", _image_show)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def plot(self):
        fig, ax = plt.subplots(1)
        ax.imshow(self.image, "gray")
        plt.show()
        plt.close(fig)

    def get_image(self):
        """Get image array."""
        return self.image


def indices_in_window(x0, y0, x1, y1):
    I, J = np.meshgrid(np.arange(x0, x1+1), np.arange(y0, y1+1))
    return np.hstack((I.reshape(-1, 1), J.reshape(-1,1)))


def read_image(filepath: Path) -> Image:
    image_array = cv2.imread(str(filepath))
    return Image(image_array)


def save_image(filepath: Path, image: Image):
    success = cv2.imwrite(str(filepath.resolve()), image.get_image())
    if not success:
        raise IOError("Failed to save image")


if __name__ == "__main__":
    filepath = Path("data/scan1.png")
    reader = read_image(filepath)
    values = reader.do_stuff()

