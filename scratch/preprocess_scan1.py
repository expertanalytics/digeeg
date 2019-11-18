import numpy as np
import cv2
from pathlib import Path
from dgimage import Image, read_image
from dgimage import Image2, read_image2
from dgutils import match_contours 



def preprocess(image):
    image.blur((5, 5), 0)
    image.threshold(150)

    structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    cv2.erode(image.image, structuring_element, dst=image.image)
    cv2.dilate(image.image, structuring_element, dst=image.image)

    # 1. Extract edges
    edges = cv2.adaptiveThreshold(image.image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, -2)

    # 2. Dilate edges
    kernel = np.ones((3, 3))
    cv2.dilate(edges, kernel, dst=edges)

    # 3. src.copyTo(smooth)
    smooth = image.image.copy()

    # 4. blur smooth img
    cv2.blur(smooth, (2, 2), dst=smooth)

    # 5 smooth.copyTo(esrc, edges)
    # src, mask, dst -> dst
    cv2.copyTo(smooth, edges, image.image)       # I think this is right


if __name__ == "__main__":
    filepath = Path("../data/scan1.png")
    image = read_image2(filepath)
    image.read_image(filepath)



    # # extract signal
    # image.bgr_to_gray()
    # image.invert()

    # image.equalise()
    # image.threshold()
    # image.show()

    # preprocess(image)
    # image.show()

    # contours = image.find_contours()
    # graph_candidates = match_contours(image.match_graph_candidate, contours)

    # image.filter_contours(graph_candidates)
    # image.invert()

    # image.draw(graph_candidates, image=image.image)
