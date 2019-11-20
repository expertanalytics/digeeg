import cv2
from dgimage import Image, read_image, save_image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import imutils
from dgutils import match_contours


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
    image.invert()
    image.threshold()


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


def extract_data(image):
    fig, ax = plt.subplots(1)

    contour_mode: int = cv2.RETR_EXTERNAL       # Retreive only the external contours
    contour_method: int = cv2.CHAIN_APPROX_TC89_L1      # Apply a flavor of the Teh Chin chain approx algo

    blur = cv2.blur(image.image, (1, 1))
    edges = cv2.Canny(blur, 100, 255)

    contours = cv2.findContours(blur, contour_mode, contour_method)
    contours = imutils.grab_contours(contours)
    contours = list(filter(lambda c: c.size > 6, contours))

    features = match_contours(matcher=image.match_graph_candidate, contours=contours)
    # image_draw = image.draw(contours, image=image.image, show=True)
    # ax.imshow(image_draw, cmap="gray")
    # plt.show()

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

    """
    # Restore colors
    image.reset_image()
    image.bgr_to_gray()
    image.resample()

    image.filter_contours(features)

    # image.gray_to_bgr()
    image_draw = image.draw(features, image.image)
    image.show(image_draw)
    """



def foo(image):
    # Reorient the image to align with axis
    features = markers(image)
    image.get_axis(features)
    image.resample()
    # image.plot()

    preprocess(image)
    print("Shape: ", image.image.shape)
    # image.plot()

    extract_data(image)



if __name__ == "__main__":
    image_directory = Path("tmp_split_images")
    # for filename in image_directory.iterdir():
    #     print(filename)
    #     image = read_image(filename)
    #     foo(image)

    # filename = "../data/scan3_sample.png"
    filename = "tmp_split_images/split4.png"
    image = read_image(filename)
    foo(image)
