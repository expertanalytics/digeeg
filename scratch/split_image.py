from dgimage import Image, read_image, save_image
from pathlib import Path
from dgutils import match_contours, get_contours, get_marker_matcher
import math

import matplotlib.pyplot as plt

import cv2
import numpy as np


def plot(image: np.ndarray):
    fig, ax = plt.subplots(1)
    ax.imshow(image, cmap="gray")
    plt.show()
    plt.close(fig)


def markers(image: Image, kernel_length: int = 5):
    """Return the contours of the black square markers."""
    image.bgr_to_gray()
    image.invert()
    image.blur(9)
    image.threshold(150)

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

    contours = get_contours(image)
    features = match_contours(matcher=get_marker_matcher(image), contours=contours)

    print("Num markers: ", len(features))
    image.reset_image()
    return features


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
    return new_image_list


if __name__ == "__main__":
    filepath = Path("../data/scan4.png")
    # filepath = Path("../data/scan3_sample.png")

    image = read_image(filepath)
    image_list = split_image(image)
    for i, image in enumerate(image_list):
        save_image(Path("tmp_split_images") / f"split{i}.png", Image(image))
