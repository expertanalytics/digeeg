import typing as tp
import numpy as np

import cv2

from dataclasses import dataclass
from pathlib import Path
from itertools import cycle

import imutils

from dgutils import contour_interior
from dgresample import Line
from .colors import colors


@dataclass
class Image:
    image: np.ndarray

    # TODO: declare image orig, but skip constructor?
    image_orig: np.ndarray = None

    # TODO: self.axis, but skip constructor?
    axis: tp.Optional[tp.Tuple[np.ndarray]] = None

    # Default resampling parameters
    resample_x_max: float = 1.8
    resample_y_max: float = 0.8
    resample_step_x: float = 1/1000
    resample_step_y: float = 1/1000

    scale: float = 1

    def __post_init__(self):
        self.reset_image()

    def reset_image(self) -> None:
        """Copy original image to image."""
        self.image_orig = self.image.copy()

    def reduce_contour(
        self,
        contour: np.ndarray,
        *,
        operation: tp.callable[[np.ndarray], float]
    ) -> float:
        I, J = contour_interior(contour).T
        return operation(self.mage[J, I])

    def image_moment(self, order: int = 1):
        image = self.image
        if len(image.shape) > 2:
            image = np.mean(image, axis=2)

        m, n = image.shape
        v = np.mean(image)
        x0 = np.mean(image * np.arange(n)**order) / v
        y0 = np.mean(image * np.arange(m)[:, np.newaxis]**order) / v
        return x0, y0

    def resample(
        self,
        axis: tp.Tuple[Line],
        scale: float,
    ):
        x_axis, y_axis = axis
        origin = x_axis ^ y_axis
        a = x_axis ^ (y_axis - scale*self.resample_x_max)
        b = (x_axis - scale*self.resample_y_max) ^ y_axis

        n_x = int(self.resample_x_max / self.resample_step_x)
        n_y = int(self.resample_y_max / self.resample_step_y)

        # Get the coordinates defining affine transform
        source_pts = np.array([origin, a, b], dtype="float32")
        target_pts = np.array([[0, n_y], [n_x, n_y], [0, 0]], dtype="float32")
        mapping = cv2.getAffineTransform(source_pts, target_pts)

        self.image = cv2.warpAffine(self.image, mapping, (n_x, n_y))        # dst = self.image?

    def draw(self, features: tp.Sequence[np.ndarray], image: Image = None) -> None:
        image_draw = image if image is not None else self.image_orig.copy()
        color_iterator = cycle(colors)
        lw = 1      # TODO: Line width?

        if self.axis is not None:
            color = next(color_iterator)
            x_axis, y_axis = self.axis

            pt0, pt1 = x_axis.get_line_segment(image_draw)
            cv2.line(image_draw, pt0, pt1, color.bgr, lw)

            pt0, pt1 = y_axis.get_line_segment(image_draw)
            cv2.line(image_draw, pt0, pt1, color.bgr, lw)

            pt0, pt1 = (y_axis - self.scale*self.resample_x_max).get_line_segment(image_draw)
            cv2.line(image_draw, pt0, pt1, color.bgr, lw)

            pt0, pt1 = (x_axis - self.scale*self.resample_y_max).get_line_segment(image_draw)
            cv2.line(image_draw, pt0, pt1, color.bgr, lw)

            x0, y0 = image_draw.image_moment()
            cv2.circle(image_draw, (int(x0), int(y0)),  5, color.bgr)
            cv2.circle(image_draw, (int(x0), int(y0)), 25, color.bgr)

        for contours in features:
            color = next(color_iterator)
            image_draw = cv2.drawContours(image_draw, contours, -2, color.bgr, lw)

        cv2.imshow("Image", image_draw)
        cv2.waitKey(0)

    def color_transform(self, color_mode: int) -> None:
        """Change colorscheme. 

        colormode can be any of

         - cv2.COLOR_BGR2GRAY
         - cv2.COLOR_GRAY2BGR

        See cv2 documentation for more options.
        """
        self.image = cv2.cvtColor(self.image, color_mode)

    def threshold(self, threshold_value: float = None) -> None:
        """Apply a fixed level threshold to each pixel.

        dst(x, y) = maxval if src(x, y) > thresh_val else 0

        thresh_val is set from the image histogram using Otsu's binarisation, assuming the image
        histogram is bimodal.

        It is recommended to blur the image before binarisation.
        """
        if threshold_value is None:
            _, self.image = cv2.threshold(self.image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, self.image = cv2.threshold(self.image, 0, 255, cv2.THRESH_BINARY, threshold_value)

    def morph(self, transform, kernel: np.ndarray = None, iterations: int = 1):
        """ Morph the image using `transform`.

        Valid transforms include:
         - cv2.MORPH_ERODE
         - cv2.MORPH_OPEN
         - cv2.MORPH_CLOSE
         - cv2.MORPH_DILATE

        kernel dafaults to np.ones((3, 3))
        """
        if kernel is None:
            _kernel = np.ones((3, 3))

        self.image = cv2.morphologyEx(
            self.image,
            transform,
            kernel=_kernel,
            iterations=iterations
        )

    def invert(self, threshold_maxvalue: int = 255) -> None:
        """Invert a binary greyscale image."""
        self.image = threshold_maxvalue - self.image

    def find_contours(
        self,
        min_size: int = 4,
        contour_mode: int = cv2.RETR_EXTERNAL,
        contour_method = cv2.CHAIN_APPROX_TC89_L1
    ) -> tp.List[np.ndarray]:
        """Find contours in a binary image."""
        contours = cv2.findContours(self.image, contour_mode, contour_method)
        contours = imutils.grab_contours(contours)
        contours = list(filter(lambda c: c.size > min_size, contours))
        return contours

    def contour_mask(self, contours: tp.Sequence[np.ndarray]) -> None:
        """Keep only the pixels inside the contours"""
        shape = (m, n) = self.image.shape[:2]
        image_filter = np.zeros(shape, dtype=self.image.dtype)

        # TODO: Why -2?
        image_filter = cv2.drawContours(image_filter, contours, -2, 255, cv2.FILLED)
        self.image = np.fmin(self.image, image_filter)

    def image_to_point_cloud(self) -> np.ndarray:
        I, x = np.where(self.image)     # TODO: Huh?
        y = self.image.shape[0] - I - 1
        return np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))


def read_image(filepath: Path) -> Image:
    image_array = cv2.imread(str(filepath))
    return Image(image_array)


def save_image(filepath: Path, image: Image):
    success = cv2.imwrite(str(filepath.resolv()), image.get_image())
    if not success:
        raise IOError("Failed to save image")
