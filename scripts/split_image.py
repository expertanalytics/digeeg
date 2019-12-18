import numpy as np
import matplotlib.pyplot as plt
import typing as tp

import cv2
import math

from pathlib import Path

from dgimage import (
    Image,
    read_image,
    save_image,
    resample,
    get_axis,
    dump_image
)

from dgutils import (
    match_contours,
    get_contours,
    get_marker_matcher,
    plot,
)


def markers(image: Image, kernel_length: int = 5):
    """Return the contours of the black square markers."""
    assert len(image.image.shape) == 2, f"Expecting binary image"
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

    contours = get_contours(image=image)
    features = match_contours(matcher=get_marker_matcher(image=image), contours=contours)
    image.reset_image()
    return features


def split_image(image):
    image.bgr_to_gray()
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
        new_image_list.append(Image(new_image))
        rectangle_indices += 1
    return new_image_list


def run(input_image_path: Path, output_directory: Path, identifier: str):
    image = read_image(input_image_path)
    image_list = split_image(image)

    scale_dict = {}
    for i, image in enumerate(image_list):
        image.bgr_to_gray()
        features = markers(image)

        axis, scale = get_axis(image, features)
        image.set_axis(axis)
        image.set_scale(scale)
        old_scale = scale
        image.reset_image()
        resample(image, step_x=2.5e-4, step_y=2.5e-4)
        image.checkpoint()

        # Recompute scale
        image.bgr_to_gray()
        features = markers(image)       # 
        axis, scale = get_axis(image, features)
        image.set_axis(axis)
        image.set_scale(scale)
        scale_dict[i] = scale

    output_directory.mkdir(exist_ok=True, parents=True)

    for i, image in enumerate(image_list):
        print("Scale: ", image.scale)
        save_image(output_directory / f"{identifier}_split{i}.png", image)
        # dump_image(output_directory / f"split{i}_{identifier}.pkl", image)

    with open(output_directory / "scales.txt", "a") as output_handle:
        for case, scale in scale_dict.items():
            output_handle.write(f"Case: {case}, Scale: {scale}\n")


if __name__ == "__main__":
    import sys
    input_image_path = Path(sys.argv[1])
    output_directory = Path(sys.argv[2])
    identifier = sys.argv[3]
    run(input_image_path, output_directory, identifier)

    # filepath = Path("../data/scan4.png")
    # identifier = "scan4"
    # run(filepath, identifier)
    # filepath = Path("../data/scan3_sample.png")
