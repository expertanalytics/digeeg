import numpy as np
import typing as tp
import matplotlib.pyplot as plt

import cv2
import shapely.geometry

from dataclasses import dataclass
from pathlib import Path
from itertools import cycle

import imutils

import collections

from dgutils import contour_interior, Line, angles_in_contour, rectangle_aspect_ratio, indices_in_window
from .colors import colors



@dataclass
class Image2:
    image_orig: np.ndarray
    image: np.ndarray = None

    # Do I need these here?
    resample_x_max: float = 1.8
    resample_y_max: float = 0.8
    resample_step_x: float = 1/1000
    resample_step_y: float = 1/1000
    scale: float = 1

    axis: tp.Tuple[np.ndarray, np.ndarray] = None

    def __post_init__(self):
        self.reset_image()

    def reset_image(self) -> None:
        """Copy original image to image."""
        self.image = self.image_orig.copy()

    def reduce_contour(
        self,
        contour: np.ndarray,
        *,
        operation: tp.Callable[[np.ndarray], float]
    ) -> float:
        I, J = contour_interior(contour).T
        return operation(self.image[J, I])

    def image_moment(self, order: int = 1):
        image = self.image
        if len(image.shape) > 2:
            image = np.mean(image, axis=2)

        m, n = image.shape
        v = np.mean(image)
        x0 = np.mean(image * np.arange(n)**order) / v
        y0 = np.mean(image * np.arange(m)[:, np.newaxis]**order) / v
        return x0, y0

    def get_axis(self, rectangles: tp.List[np.ndarray]):
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
                B[4*i + j, int(centreline <= pt)] = 1

        svd = np.linalg.svd(B)
        c, d, a, b = svd[-1][-1, :]

        # Equation for x-axis -- best fit for centreline
        # x_axis = Line(a, b, (c + d)/2)
        x_axis = Line(a, b, c)      # Why not (c + d) / 2

        # Get image gentroid and orient the line
        image_centre = self.image_moment(order=1)
        if x_axis >= image_centre:
            x_axis = -x_axis

        # Place a preliminary y-axis
        y_axis_prelim = x_axis.orthogonal_line(image_centre)

        # Place origin on the first marker along the oriented x-axis
        origin = sorted(centres, key=y_axis_prelim)[0]
        y_axis = x_axis.orthogonal_line(origin)

        self.axis = (x_axis, y_axis)
        self.scale = np.linalg.norm(centres[1] - centres[0])

    def match_contours(self,
                       match_types: tp.List[str],
                       contours: tp.List[np.ndarray] = None):
        """Iterate over the contours and apply a classifying function to each contour.

        Return a dictionary of the matches with the match type as key. There can only
        be one matchtype per contour.
        """
        contours = contours or self.find_contours()

        matchers = {type: getattr(self, f"match_{type}") for type in match_types}
        matches = collections.defaultdict(list)

        for c in contours:
            for match_type, matcher in matchers.items():
                match_result = matcher(c)
                if match_result is not None:
                    matches[match_type].append(match_result)
                    break

        return matches

    def match_rectangle(self, c: np.ndarray):
        """If `c` can be approximated by a square, return an approximation."""
        perimeter = cv2.arcLength(c, True)

        # create a closed polygonal approximation to `c` with the distance between them
        # less than `epsilon*perimeter`.
        approx = cv2.approxPolyDP(c, 0.04*perimeter, True)

        if len(approx) == 4:
            angles = angles_in_contour(approx)
            # Check angles for rectangle
            if max(abs(angles - np.pi/2)) < 0.1  * np.pi:
                return approx

    def match_marker(self, c: np.ndarray) -> np.ndarray:
        """Return rectange if it matches a square marker on the paper strip."""
        rectangle = self.match_rectangle(c)

        if (rectangle is not None
                and rectangle_aspect_ratio(rectangle) > 0.5
                and self.get_contour_area(rectangle) > 100
                and self.get_contour_mean_value(rectangle) < 100):
            return rectangle

    def match_graph_candidate(self, c: np.ndarray) -> np.ndarray:
        """Return contour if it is poorly approximated by a circle."""
        perimeter = cv2.arcLength(c, True)
        pg = shapely.geometry.Polygon(c.reshape(-1, 2))
        rel_area = 4 * pg.area / perimeter**2
        if rel_area < 0.1:
            return c

    def get_contour_area(self, c):
        return cv2.contourArea(c)
        # pg = shapely.geometry.Polygon(c.reshape(-1, 2))
        # return pg.area

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

    def load_image(self, filepath: Path) -> None:
        self.filepath = filepath
        self.image_orig = cv2.imread(str(filepath))
        self.reset_image()      # Copy image
        self.axis = None
        self.scale = None

    def bgr_to_gray(self) -> None:
        """Convert image to greyscale."""
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def gray_to_bgr(self):
        self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)

    def threshold(self, thresh_val: float = None) -> None:
        """Apply a fixed level threshold to each pixel.

        dst(x, y) = maxval if src(x, y) > thresh_val else 0

        thresh_val is set from the image histogram using Otsu's binarisation, assuming the image
        histogram is bimodal.

        It is recommended to blur the image before binarisation.
        """
        if thresh_val:
            _, self.image = cv2.threshold(self.image, thresh_val, 255, cv2.THRESH_BINARY)
        else:
            _, self.image = cv2.threshold(self.image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    def blur(self, kernel_size: tp.Tuple[int, int], sigma: float):
        self.image = cv2.GaussianBlur(self.image, kernel_size, sigma)

    def morph(self, transform, kernel=None, iterations=None):
        """
        Valid transforms include:
         - cv2.MORPH_ERODE
         - cv2.MORPH_OPEN
         - cv2.MORPH_CLOSE
         - cv2.MORPH_DILATE

        """
        if kernel is None:
            kernel = self.morph_kernel
        iterations = iterations or self.morph_num_iter
        self.image = cv2.morphologyEx(
            self.image,
            transform,
            kernel=kernel,
            iterations=iterations
        )

    def close(self, kernel=None, iterations=None):
        if kernel is None:
            kernel = self.morph_kernel

        if iterations is None:
            iterations = self.morph_num_iter

    def invert(self) -> None:
        """Invert a binary greyscale image."""
        # TODO: How does invert work with thresh maxval
        self.image = 255 - self.image

    def equalise(self):
        self.image = cv2.equalizeHist(self.image)       # Works wonders for low quality?

    def find_contours(
        self,
        min_size: int = 6,
        contour_mode: int = cv2.RETR_EXTERNAL,
        contour_method = cv2.CHAIN_APPROX_TC89_L1
    ) -> tp.List[np.ndarray]:
        """Find contours in a binary image."""
        contours = cv2.findContours(self.image, contour_mode, contour_method)
        contours = imutils.grab_contours(contours)
        contours = list(filter(lambda c: c.size > min_size, contours))
        return contours

    def set_axis(self, xaxis: np.ndarray, yaxis: np.ndarray):
        self.axis = xaxis, yaxis

    def filter_contours(self, contours):
        """Keep only the pixels inside the contours """

        shape = (m, n) = self.image.shape[:2]
        image_filter = np.zeros(shape, dtype=self.image.dtype)
        image_filter = cv2.drawContours(image_filter, contours, -2, 255, cv2.FILLED)

        self.image = np.fmin(self.image, image_filter)

    def resample(self):
        x_axis, y_axis = self.axis
        origin = x_axis ^ y_axis
        a = x_axis ^ (y_axis - self.scale * self.resample_x_max)
        b = (x_axis - self.scale * self.resample_y_max) ^ y_axis

        n_x = int(self.resample_x_max / self.resample_step_x)
        n_y = int(self.resample_y_max / self.resample_step_y)

        # Get the coordinates defining affine transform
        source_pts = np.array([origin, a, b], dtype="float32")
        target_pts = np.array([[0,n_y], [n_x, n_y], [0, 0]], dtype="float32")
        mapping = cv2.getAffineTransform(source_pts, target_pts)

        cv2.warpAffine(self.image, mapping, (n_x, n_y), dst=self.image)

    def image_to_point_cloud(self):
        I, x = np.where(self.image)
        y = self.image.shape[0] - I - 1
        return np.hstack((x.reshape(-1,1), y.reshape(-1,1)))

    def draw(self, features, image=None):
        image_draw = image if image is not None else self.image_orig.copy()
        color_iterator = cycle(colors)
        lw = 1

        if self.axis:
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

            x0, y0 = self.image_moment()
            cv2.circle(image_draw, (int(x0), int(y0)),  5, color.bgr)
            cv2.circle(image_draw, (int(x0), int(y0)), 25, color.bgr)

        for contours in features:
            color = next(color_iterator)
            print(contours)
            image_draw = cv2.drawContours(image_draw, contours, -2, color.bgr, lw)

        cv2.imshow("Image", image_draw)
        cv2.waitKey(0)

    def plot(self, image=None):
        fig, ax = plt.subplots(1)
        if image is None:
            _image = self.image
        else:
            _image = image

        ax.imshow(_image, "gray")
        plt.show()
        plt.close(fig)

    def show(self):
        cv2.imshow("Current image", self.image)
        cv2.waitKey(0)

    def read_image(self, filepath: Path):
        self.load_image(filepath)
        self.show()

        # Try to find the black markers in a fairly sharp image
        # First convert to binary grayscale image and convert
        self.bgr_to_gray()
        self.invert()
        self.blur((3, 3), 0)

        # self.image = cv2.equalizeHist(self.image)       # Works wonders for low quality?

        self.threshold()

        horizontal = self.image.copy()

        structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
        horizontal = cv2.erode(horizontal, structuring_element)
        horizontal = cv2.dilate(horizontal, structuring_element)

        # 1. Extract edges
        edges = cv2.adaptiveThreshold(horizontal, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, -2)

        # 2. Dilate edges
        kernel = np.ones((2, 2))
        edges = cv2.dilate(edges, kernel)

        # 3. src.copyTo(smooth)
        smooth = horizontal.copy()

        # 4. blur smooth img
        smooth = cv2.blur(smooth, (2, 2))

        # 5 smooth.copyTo(src, edges)
        # src, mask, dst -> dst
        cv2.copyTo(smooth, horizontal, edges)       # I think this is right

        self.plot(horizontal)
        self.image = smooth.copy()

        # self.morph(cv2.MORPH_CLOSE)   # Don't think I need this for nice images
        # self.plot()

        features = self.match_contours(match_types=["marker"])

        # Now we can find axes
        markers = features["marker"]
        self.get_axis(markers)

        # Reset and find graphs
        self.reset_image()
        self.bgr_to_gray()
        self.invert()
        # self.blur(3)
        self.threshold()

        features.update(self.match_contours(match_types=["graph_candidate"]))
        graphs = features["graph_candidate"]


        # Reset and keep only graphs
        self.reset_image()
        self.bgr_to_gray()
        self.invert()
        self.filter_contours(graphs)
        self.invert()

        # Restore colors
        self.gray_to_bgr()
        self.draw(graphs, image=self.image)

        # self.reset_image()
        # self.bgr_to_gray()
        # self.invert()
        # self.filter_contours(graphs)
        # self.resample()
        # self.blur()
        # self.invert()
        # self.show()

        # self.invert()
        # return self.image_to_point_cloud()

        # self.draw(features)
        # return features


def read_image2(filepath: Path) -> "Image":
    image_array = cv2.imread(str(filepath))
    return Image2(image_array)


def save_image(filepath: Path, image: "Image"):
    success = cv2.imwrite(str(filepath.resolv()), image.get_image())
    if not success:
        raise IOError("Failed to save image")
