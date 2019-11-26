import typing as tp
from pathlib import Path
import dataclasses
import collections
import itertools

import operator

import cv2
import imutils
import numpy as np

import shapely.geometry

import matplotlib.pyplot as plt

from .line import Line

from .colors import colors

from .image_utils import get_image_moment, color_to_grayscale, grayscale_to_color


@dataclasses.dataclass
class Image:
    image_orig: np.ndarray
    image: np.ndarray = None

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

    def reset_image(self, tag: str = None) -> None:
        """Restore image from checkpoint.

        If `tag` does not exist, resotre `image_orig`.
        """
        self.image = self.checkpoint_dict.get(tag, self.image_orig).copy()

    def bgr_to_gray(self) -> None:
        """Convert image to greyscale."""
        self.image = color_to_grayscale(self.image)

    def gray_to_bgr(self) -> None:
        self.image = grayscale_to_color(self.image)

    def threshold(self, thresh_val: float = None) -> None:
        """Apply a fixed level threshold to each pixel.

        dst(x, y) = maxval if src(x, y) > thresh_val else 0

        thresh_val is set from the image histogram using Otsu's binarisation, assuming the image
        histogram is bimodal.

        It is recommended to blur the image before binarisation.
        """
        if thresh_val is None:
            cv2.threshold(self.image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU, dst=self.image)
        else:
            cv2.threshold(self.image, thresh_val, 255, cv2.THRESH_BINARY, dst=self.image)

    def gaussian_blur(self, kernel_size: int) -> None:
        """Convolve the image with a zero mean gaussian kernel."""
        cv2.GaussianBlur(self.image, (kernel_size, kernel_size), 0, dst=self.image)

    def blur(self, kernel_size: int) -> None:
        """Blur the image using a normalised box filter."""
        cv2.blur(self.image, (kernel_size, kernel_size), dst=self.image)

    def morph(self, transform: int, kernel_size: tp.Tuple[int, int], iterations: int):
        """Apply a geometric transform to the image.

        NB! The foregprund should be white!

        Valid transforms include:
         - cv2.MORPH_ERODE
         - cv2.MORPH_OPEN
         - cv2.MORPH_CLOSE
         - cv2.MORPH_DILATE
        """
        kernel = np.ones(kernel_size)
        cv2.morphologyEx(
            self.image,
            transform,
            kernel=kernel,
            iterations=iterations,
            dst=self.image
        )

    def invert(self, max_value=255) -> None:
        """Invert a binary greyscale image."""
        self.image *= np.uint8(-1)
        self.image += np.uint8(max_value)

    def set_axis(self, axis: tp.Tuple[np.ndarray, np.ndarray]) -> None:
        self.axis = axis

    def set_scale(self, scale: float) -> None:
        self.scale = scale

    def draw(self, features, draw_axis=True, show=True):
        """Draw the image with overlaid contours."""
        color_iterator = itertools.cycle(colors)
        lw = 1

        image_draw = self.copy_image()
        if len(image_draw.shape) < 3:
            image_draw = grayscale_to_color(image_draw)

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

            x0, y0 = get_image_moment(self.image, order=1)
            cv2.circle(image_draw, (int(x0), int(y0)),  5, color.bgr)
            cv2.circle(image_draw, (int(x0), int(y0)), 25, color.bgr)

        color = next(color_iterator)
        image_draw = cv2.drawContours(image_draw, features, -2, color.bgr, lw)

        if show:
            cv2.imshow("Image", image_draw)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return image_draw

def read_image(filepath: Path) -> Image:
    image_array = cv2.imread(str(filepath))
    return Image(image_array)


def save_image(filepath: Path, image: Image):
    success = cv2.imwrite(str(filepath.resolve()), image.image)
    if not success:
        raise IOError("Failed to save image")


if __name__ == "__main__":
    filepath = Path("data/scan1.png")
    reader = read_image(filepath)
    values = reader.do_stuff()

