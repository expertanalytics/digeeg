from dgimage import Image
from pathlib import Path
from scipy import interpolate
from scipy import stats
from dgutils import match_contours

import matplotlib.pyplot as plt

import imutils

import cv2
import numpy as np


def preprocess(image):
    image.bgr_to_gray()
    image.invert()
    image.blur(3)
    image.threshold()

    image.blur(2)
    structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cv2.erode(image.image, structuring_element, dst=image.image)
    cv2.dilate(image.image, structuring_element, dst=image.image)

    # 1. Extract edges
    edges = cv2.adaptiveThreshold(image.image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, -2)

    # 2. Dilate edges
    kernel = np.ones((2, 2))
    edges = cv2.dilate(edges, kernel)

    # 4. blur smooth img
    smooth = cv2.blur(image.image, (2, 2))

    # 5 smooth.copyTo(src, edges)
    # src, mask, dst -> dst
    cv2.copyTo(smooth, image.image, edges)       # I think this is right


def markers(image):
    features = image.match_contours(match_types=["marker"])
    image.get_axis(features["marker"])
    image.save_image(Path("tmp/marked1.png"))
    # image.show(mode=cv2.WINDOW_NORMAL)
    # image.reasmple_x_max = 10
    # image.reasmple_y_max = 3
    image.resample()
    image.save_image(Path("tmp/marked2.png"))
    # image.gray_to_bgr()
    # image.show(mode=cv2.WINDOW_NORMAL)
    image.plot()


def graphs(image):
    contour_mode: int = cv2.RETR_EXTERNAL       # Retreive only the external contours
    contour_method: int = cv2.CHAIN_APPROX_TC89_L1      # Apply a flavor of the Teh Chin chain approx algo

    blur = cv2.blur(image.image, (1, 1))
    edges = cv2.Canny(blur, 100, 255)

    contours = cv2.findContours(blur, contour_mode, contour_method)
    contours = imutils.grab_contours(contours)
    contours = list(filter(lambda c: c.size > 6, contours))

    features = match_contours(matcher=image.match_graph_candidate, contours=contours)

    image_copy = image.image.copy()
    image.gaussian_blur(5)
    image.threshold()
    features = match_contours(matcher=image.match_graph_candidate, contours=contours)
    image.filter_contours(features)
    image.blur(3)
    contours = cv2.findContours(blur, contour_mode, contour_method)
    contours = imutils.grab_contours(contours)
    contours = list(filter(lambda c: c.size > 6, contours))
    features = match_contours(matcher=image.match_graph_candidate, contours=contours)

    # Restore colors
    image.reset_image()
    image.bgr_to_gray()
    # image.resample()
    image.filter_contours(features)

    # image.gray_to_bgr()
    image_draw = image.draw(features, image.image)
    image.show(image_draw)


def extract(image):
    print(image.image.shape)
    fig, (ax1, ax2) = plt.subplots(2)

    plot_img = image.image.copy()
    plot_img[:, 399:402] = 0

    print(plot_img)
    ax1.imshow(plot_img)

    values = image.image[:, 400]
    x = np.linspace(0, 1, values.size)
    t, c, k = interpolate.splrep(x, values, s=10, k=3)

    xnew = np.linspace(0, 1, 10000)
    splev = interpolate.BSpline(t, c, k)

    ax2.plot(x, values, "--", alpha=0.75)
    ax2.plot(xnew, splev(xnew))
    plt.show()


if __name__ == "__main__":
    image = Image()
    filepath = Path("../data/scan4.png")
    image.load_image(filepath)
    # plt.imshow(image.image)
    # image.show()

    preprocess(image)
    markers(image)
    # graphs(image)

    # extract(image)
