from dgimage import Image, read_image
from pathlib import Path
from scipy import interpolate
from scipy import stats
from dgutils import match_contours
import operator
import itertools
import math

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
    image.bgr_to_gray()
    image.invert()
    image.blur(9)
    image.threshold(150)

    kernel_length = 5
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    horisontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))

    # vertial lines
    vertical_image = cv2.erode(image.image, vertical_kernel, iterations=4)
    cv2.dilate(vertical_image, vertical_kernel, iterations=4, dst=vertical_image)

    # Horisontal lines
    horisontal_image = cv2.erode(image.image, horisontal_kernel, iterations=4)
    cv2.dilate(image.image, horisontal_kernel, iterations=4, dst=vertical_image)

    # Compute intersection of horisontal and vertical
    cv2.bitwise_and(horisontal_image, vertical_image, dst=image.image)

    features = image.match_contours(match_types=["marker"])
    print("Num markers: ", len(features["marker"]))
    image.reset_image()
    return features["marker"]

    # image.get_axis(features["marker"])
    # image.resample()
    # image.plot()


def split_image(image):
    rectangles = markers(image)

    rectangle_array = np.zeros((len(rectangles), 4, 2))
    for i, rec in enumerate(rectangles):
        for n, vertex in enumerate(rec):
            rectangle_array[i, n, :] = vertex[0]

    centres = np.mean(rectangle_array, axis=1)
    max_dx = np.max(centres[:, 0])
    max_dy = np.max(centres[:, 1])

    new_image_list = []

    horisontal_image = max_dx >= max_dy
    if horisontal_image:
        sorted_indices = np.argsort(centres[:, 0])
        sorted_rectangle_vertices = rectangle_array[sorted_indices, :, 0]
    else:
        sorted_indices = np.argsort(centres[:, 1])
        sorted_rectangle_vertices = rectangle_array[sorted_indices, :, 1]

    rectangle_indices = np.arange(2)
    for i in range(sorted_rectangle_vertices.shape[0] - 1):
        square1, square2 = sorted_rectangle_vertices[rectangle_indices]
        min_index = math.floor(min(map(np.min, (square1, square2))))
        max_index = math.ceil(max(map(np.max, (square1, square2))))

        if i == 0:
            min_index = 0
        if i == sorted_rectangle_vertices.shape[0] - 2:
            max_index = None        # include the last index

        if horisontal_image:
            new_image = image.image[:, min_index:max_index]
        else:
            new_image = image.image[min_index:max_index, :]
        new_image_list.append(new_image)
        rectangle_indices += 1

    for image in new_image_list:
        plt.imshow(image)
        plt.show()
        plt.close()
    return new_image_list


if __name__ == "__main__":
    # image = Image()
    # filepath = Path("../data/scan4.png")
    filepath = Path("../data/scan3_sample.png")
    # image.load_image(filepath)
    image = read_image(filepath)
    split_image(image)



    # markers(image)
    # plt.imshow(image.image)
    # image.show()

    # preprocess(image)
    # image.plot()
