import numpy as np
import matplotlib.pyplot as plt
import typing as tp

import cv2
import math
import argparse
import logging
import os

from pathlib import Path

from dgimage import (
    Image,
    read_image,
    save_image,
    resample,
    get_axis,
    dump_image,
)

from dgutils import (
    plot,
    markers,
    save,
    get_debug_path,
)


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


logger = logging.getLogger(__name__)


def split_image(
    image: Image,
    *,
    kernel_length: int = 5,
    blur_kernel_size: int = 9,
    threshold_value: float = 150,
    num_iterations: int = 4
) -> tp.List[Image]:
    """Split an EEG scan into manageable parts.

    Detect the black square fiduciary markers and split the scan into parts such that each
    part contains two such markers.

    kernel_length:
        Govern how aggressive to be when removing horisontal and vertical background structures.
    blur_kernel_size:
        The size of the blur kernel for the initial blur then threshold operation.
    threshold_value:
        Thresholding is done right after the blurring.
    num_iterations:
        Number of iterations in the erode and dilate transformations.
    """
    image.bgr_to_gray()
    rectangles = markers(
        image,
        kernel_length=kernel_length,
        blur_kernel_size=blur_kernel_size,
        threshold_value=threshold_value,
        num_iterations=num_iterations,
    )

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


def run(
    input_image_path: Path,
    output_directory: Path,
    name: str,
    *,
    debug: bool = False,
    compute_scale: bool = True,
    kernel_length: int = 9,
    num_iterations: int = 4,
    blur_size: int = 9,
    thresh_val: float = 150,
    dx: float = 2.5e-4,
    dy: float = 2.5e-4
):
    image = read_image(input_image_path)
    image_list = split_image(image)

    if debug:
        debug_path = get_debug_path("split_image")
        save(image.image, debug_path, "input")

    scale_dict = {}
    for i, image in enumerate(image_list):
        image.bgr_to_gray()
        # TODO: debug each split
        rectangles = markers(
            image,
            kernel_length=kernel_length,
            blur_kernel_size=blur_size,
            threshold_value=thresh_val,
            num_iterations=num_iterations,
            debug=debug
        )

        axis, scale = get_axis(image, rectangles)
        image.set_axis(axis)
        image.set_scale(scale)
        image.reset_image()
        resample(image, step_x=dx, step_y=dy)
        image.checkpoint()

        if compute_scale:    # Recompute scale
            image.bgr_to_gray()
            rectangles = markers(
                image,
                kernel_length=kernel_length,
                blur_kernel_size=blur_size,
                threshold_value=thresh_val,
                num_iterations=num_iterations,
                debug=debug
            )
            axis, scale = get_axis(image, rectangles)
            image.set_axis(axis)
            image.set_scale(scale)
            scale_dict[i] = scale

    output_directory.mkdir(exist_ok=True, parents=True)

    for i, image in enumerate(image_list):
        print("Scale: ", image.scale)
        save_image(output_directory / f"{name}_split{i}.png", image)
        # dump_image(output_directory / f"split{i}_{identifier}.pkl", image)

    with open(output_directory / "scales.txt", "a") as output_handle:
        for case, scale in scale_dict.items():
            output_handle.write(f"Case: {case}, Scale: {scale}\n")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split an paper strip based on the black squares"
    )
    parser.add_argument(
        "-i",
        "--input",
        help="Path to original image",
        type=Path,
        required=True
    )

    parser.add_argument(
        "-o",
        "--odir",
        help="utput directory",
        type=Path,
        required=True
    )

    parser.add_argument(
        "-n",
        "--name",
        help="Base name of output files.",
        required=True
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Store intermediate images for calibration.",
        required=False
    )

    parser.add_argument(
        "--compute_scale",
        action="store_true",
        help="Compute the scaling based on feduciary markers.",
        required=False
    )

    parser.add_argument(
        "--kernel_length",
        type=int,
        default=5,
        help="Length of morphological kernel for removing everything but square fiducairies.",
    )

    parser.add_argument(
        "--num_iterations",
        type=int,
        default=4,
        help="Number of iterations in open/close morphological transformations.",
    )

    parser.add_argument(
        "--blur_size",
        type=int,
        default=9,
        help="Blur kernel size. Used in conjunction with `threshold`.",
    )

    parser.add_argument(
        "--thresh_val",
        type=float,
        default=150,
        help="Apply thresholding to binarise image. Use '-1' for Otsu's binarisation.",
    )

    parser.add_argument(
        "--dx",
        type=float,
        default=2.5e-4,
        help="scaling of new x-axis. See `dgimage.resample` for more detail.",
    )

    parser.add_argument(
        "--dy",
        type=float,
        default=2.5e-4,
        help="Scaling of new y-axis. See `dgimage.resample` for more detail.",
    )
    return parser


if __name__ == "__main__":
    parser = create_parser()
    cmd_args = parser.parse_args()

    args = (cmd_args.input, cmd_args.odir, cmd_args.name)
    kwargs = {
        "debug": cmd_args.debug,
        "compute_scale": cmd_args.compute_scale,
        "kernel_length": cmd_args.kernel_length,
        "num_iterations": cmd_args.num_iterations,
        "blur_size": cmd_args.blur_size,
        "thresh_val": cmd_args.thresh_val,
        "dx": cmd_args.dx,
        "dy": cmd_args.dy
    }
    run(*args, **kwargs)
