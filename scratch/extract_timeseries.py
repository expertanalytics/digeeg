import cv2
from dgimage import Image, read_image, save_image
from pathlib import Path
import matplotlib.pyplot as plt
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


def foo(image):
    features = markers(image)
    print(image.image.shape)

    image.get_axis(features)
    image.plot()
    image.resample()
    image.plot()



if __name__ == "__main__":
    image_directory = Path("tmp_split_images")
    for filename in image_directory.iterdir():
        filename = "../data/scan3_sample.png"
        print(filename)
        image = read_image(filename)
        foo(image)
        break
