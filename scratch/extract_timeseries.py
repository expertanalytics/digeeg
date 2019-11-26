import cv2
from dgimage import Image, read_image, save_image, resample, get_axis
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import imutils

from dgutils import (
    match_contours,
    filter_contours,
    remove_contours,
    filter_image,
    get_square_matcher,
    get_bounding_rectangle_matcher,
    get_contours,
    get_marker_matcher,
    get_graph_matcher,
)


def plot(image_array):
    fig, ax = plt.subplots(1)
    ax.imshow(image_array, cmap="gray")
    plt.show()
    plt.close(fig)


def preprocess(image):
    image.invert()
    image.blur(2)
    image.threshold(100)

    structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cv2.erode(image.image, structuring_element, dst=image.image)
    cv2.dilate(image.image, structuring_element, dst=image.image)

    # 1. Extract edges
    edges = cv2.adaptiveThreshold(image.image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, -2)

    # 2. Dilate edges
    kernel = np.ones((3, 3))
    edges = cv2.dilate(edges, kernel)

    # 4. blur smooth img
    smooth = cv2.blur(image.image, (3, 3))
    # 5 smooth.copyTo(src, edges)
    # src, mask, dst -> dst
    cv2.copyTo(smooth, image.image, edges)       # I think this is right
    image.invert()
    image.checkpoint("preprocessed")


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

    contours = get_contours(image)
    features = match_contours(matcher=get_marker_matcher(image), contours=contours)

    print("Num markers: ", len(features))
    return features
    # image.plot()


def extract_data2(image):
    contour_mode: int = cv2.RETR_EXTERNAL       # Retreive only the external contours
    contour_method: int = cv2.CHAIN_APPROX_TC89_L1      # Apply a flavor of the Teh Chin chain approx algo

    ######################################################################################
    ### Remove initial guess at contours. This should leave the text blocks.
    ######################################################################################
    image.invert()

    contours = get_contours(image)
    features = match_contours(matcher=get_graph_matcher(), contours=contours)
    remove_contours(image, features)

    #####################################################################################
    ### Remove all the remaining text blobs
    #####################################################################################

    image.blur(9)
    image.threshold(100)
    image.morph(cv2.MORPH_DILATE, (3, 3), 3)

    contours = get_contours(image, min_size=5)

    # Compute all the bounding boxes and filter based on the aspect ratio?
    features = match_contours(
        matcher=get_bounding_rectangle_matcher(),
        contours=contours
    )

    filter_contours(image, features)
    image.morph(cv2.MORPH_DILATE, (3, 3), 3)
    image_mask = image.copy_image()

    image.reset_image("preprocessed")
    filter_image(image, image_mask == 255)
    image.checkpoint("preprocessed")

    ######################################################################################
    ### Match the remaining graph candidates, and remove everything else
    ######################################################################################

    image.invert()
    image.blur(3)
    image.threshold()

    image.morph(cv2.MORPH_DILATE, (3, 3), 2)

    contours = get_contours(image)

    features = match_contours(matcher=get_graph_matcher(), contours=contours)
    filter_contours(image, features)
    image.invert()

    image.reset_image("resampled")
    image.draw(features)


def extract_data(image):
    image.invert()
    contour_mode: int = cv2.RETR_EXTERNAL       # Retreive only the external contours
    contour_method: int = cv2.CHAIN_APPROX_TC89_L1      # Apply a flavor of the Teh Chin chain approx algo

    contours = get_contours(image)
    features = match_contours(matcher=get_graph_matcher(), contours=contours)

    filter_contours(image, features)

    image.morph(cv2.MORPH_DILATE, (3, 3), 2)
    image_mask = image.copy_image()

    filter_image(image, image_mask == 255)

    image_filter = image.copy_image()

    image.invert()
    image.morph(cv2.MORPH_DILATE, (2, 2), 2)

    contours = get_contours(image)

    # Compute all the bounding boxes and filter based on the aspect ratio?
    features = match_contours(
        matcher=get_bounding_rectangle_matcher(),
        contours=contours
    )
    filter_contours(image, features)
    image_mask = image.copy_image()
    image.reset_image()
    filter_image(image, image_mask == 255)

    image.invert()
    image.morph(cv2.MORPH_DILATE, (3, 3), 2)

    contours = get_contours(image)

    features = match_contours(matcher=get_graph_matcher(), contours=contours)
    filter_contours(image, features)
    image.invert()

    image.reset_image()
    image_draw = image.draw(features)


def foo(image):
    # Reorient the image to align with axis
    features = markers(image)
    axis, scale = get_axis(image, features)
    image.set_axis(axis)
    image.set_scale(scale)

    image.reset_image()
    resample(image)

    image.bgr_to_gray()
    image.checkpoint("resampled")

    preprocess(image)

    extract_data2(image)
    # extract_data(image)


if __name__ == "__main__":
    image_directory = Path("tmp_split_images")
    # for filename in image_directory.iterdir():
    #     print(filename)
    #     image = read_image(filename)
    #     foo(image)

    # filename = "../data/scan3_sample.png"
    filename = "tmp_split_images/split1.png"
    image = read_image(filename)

    foo(image)
    # remove_text(image)
