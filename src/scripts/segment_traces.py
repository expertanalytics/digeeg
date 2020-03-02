import matplotlib.pyplot as plt
import numpy as np
import typing as tp

import argparse
import cv2
import math
import itertools
import os
import logging

from pathlib import Path
from shapely.geometry import Polygon

from dgimage import (
    Image,
    read_image,
    save_image,
    resample,
    get_axis,
    load_image
)

from dgutils import (
    match_contours,
    filter_contours,
    remove_contours,
    filter_image,
    get_bounding_rectangle_matcher,
    get_contours,
    get_square_matcher,
    get_marker_matcher,
    get_graph_matcher,
    show,
    plot,
    save,
    get_debug_path,
)

from dgimage import (
    colors,
    mtableau_brg,
    mbase_brg,
    mcss_brg,
    color_to_256_BRG,
    color_to_256_RGB,
)

from dgutils import remove_structured_background


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


def remove_background(
    *,
    image: Image,
    smooth_kernel_size: int = 3,
    threshold_value: int = 90,
    background_kernel_size: int = 5,
    debug: bool = False
) -> None:
    """Remove the background milimeter pattern."""
    if debug:
        debug_path = get_debug_path("remove_background")
        save(image.image, debug_path, "input")

    image.invert()              # We want the main features to be white
    image.blur(smooth_kernel_size)
    image.threshold(threshold_value)        # Removes most of the milimiter markers

    remove_structured_background(
        image_array=image.image,
        background_kernel_size=(background_kernel_size, background_kernel_size),
        smoothing_kernel_size=(smooth_kernel_size, smooth_kernel_size),
        debug=debug
    )
    if debug:
        save(image.image, debug_path, "output")

    image.invert()
    image.checkpoint("preprocessed")


def run(
    *,
    input_image_path: Path,
    output_directory: Path,
    identifier: str,
    smooth_kernel_size: int = 3,
    threshold_value: int = 100,
    background_kernel_size=5,
    scale: float = None,
    debug: bool = False,
    blue_color_filter: tp.Sequence[int] = None,
    red_color_filter: tp.Sequence[int] = None,
    horisontal_kernel_length: int = None,
    x_interval: tp.Tuple[int, int] = None
):
    """Remove the background, segment the contours and save the segmented lines as pngs."""
    image = read_image(input_image_path)

    if debug:
        debug_path = get_debug_path("prepare_lines")
        save(image.image, debug_path, "input")

    if blue_color_filter is not None:
        lower = tuple(map(int, blue_color_filter[:3]))
        upper = tuple(map(int, blue_color_filter[3:]))
        print(lower, upper)
        blue_mask = cv2.inRange(image.image, lower, upper)
        image.image[blue_mask == 255] = 255
    if red_color_filter is not None:
        lower = tuple(map(int, red_color_filter[:3]))
        upper = tuple(map(int, red_color_filter[3:]))
        red_mask = cv2.inRange(image.image, lower, upper)
        image.image[red_mask == 255] = 255

    image.invert()
    image.bgr_to_gray()
    image.checkpoint("resampled")

    if horisontal_kernel_length is not None:
        horisontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horisontal_kernel_length, 1))

        x0, x1 = x_interval
        if x1 == -1:        # Set -1 to be inclusive last element
            x1 = None
        detected_lines = cv2.morphologyEx(
            image.image[x0:x1, :],
            cv2.MORPH_OPEN,
            horisontal_kernel,
            iterations=1
        )

        # findContours returns (contours, hierarchy)
        contours = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        for c in contours:
            cv2.drawContours(image.image[x0:x1, :], [c], -1, 0, -10)

    remove_background(
        image=image,
        smooth_kernel_size=smooth_kernel_size,
        threshold_value=threshold_value,
        background_kernel_size=background_kernel_size,
        debug=debug
    )
    image.checkpoint("remove_background")

    if debug:
        save(image.image, debug_path, "match_contours")
    contours = get_contours(image=image)
    features = match_contours(matcher=get_graph_matcher(), contours=contours)

    quality_control_image = image.draw(features, show=False)
    cv2.imwrite(str(output_directory / f"QC_{identifier}.png"), quality_control_image)

    if debug:
        save(image.image, debug_path, "remove_background")

    # image.reset_image("resampled")
    image.reset_image("remove_background")
    # image.invert()

    output_directory.mkdir(exist_ok=True, parents=True)

    for i, c in enumerate(features):
        tmp_image = image.copy_image()
        filter_contours(image_array=tmp_image, contours=[c])
        clipped_contour = tmp_image[~np.all(tmp_image == 0, axis=1)]
        save_image(output_directory / f"{identifier}_trace{i}.png", Image(clipped_contour))

    ############################
    ### Make annotated image ###
    ############################

    fig, ax = plt.subplots(1, figsize=(15, 10), dpi=500)
    tmp_image = image.image_orig

    color_iterator = itertools.cycle(mtableau_brg())

    for i, c in enumerate(features):
        color = next(color_iterator)
        tmp_image = cv2.drawContours(tmp_image, features, i, color_to_256_RGB(color), cv2.FILLED)
        polygon = Polygon(c.reshape(-1, 2))
        x0, y0, x1, y1 = polygon.bounds

        ann = ax.annotate(
            f"Contour {i}",
            xy=((x0 + x1)/2, y1),        # (x0, y1)
            size=5,
            color=color,
            arrowprops=dict(
                arrowstyle='->',
                connectionstyle="arc3,rad=-0.5",
                color=color
            )
        )

    ax.imshow(tmp_image)
    ax.set_title("A digitised paper strip")
    if scale is not None:       # multiply by distance between black sqaures?
        ax.set_xticklabels(["{:.1f} cm".format(15*i/scale) for i in ax.get_xticks()])
        ax.set_yticklabels(["{:.1f} cm".format(15*i/scale) for i in ax.get_yticks()])
    ax.set_ylabel("Voltage")
    ax.set_xlabel("Time")
    fig.savefig(output_directory / f"{identifier}_annotated.png", dpi=500)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Segment EEG traces")
    parser.add_argument(
        "-i",
        "--input",
        help="Path to input image",
        type=Path,
        required=True
    )

    parser.add_argument(
        "-o",
        "--output",
        help="Path to output directory",
        type=Path,
        required=True
    )

    parser.add_argument(
        "-n",
        "--name",
        help="Identifier of the segmented trace",
        type=str,
        required=True
    )

    parser.add_argument(
        "--smooth-kernel-size",
        help="The size of the blur/smooth kernel",
        default=3,
        type=int,
        required=False
    )

    parser.add_argument(
        "--threshold-value",
        help="The threshold value used for binarisation in conjunction with smoothing",
        default=90,
        type=int,
        required=False
    )

    parser.add_argument(
        "--background-kernel-size",
        help="The size of the kernel for the morphological operations used to remove the background",
        default=5,
        type=int,
        required=False
    )

    parser.add_argument(
        "--scale",
        help="The scale of the image. What is the phusical size of a pixel.",
        default=None,
        type=float,
        required=False
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Store intermediate images for calibration.",
        required=False
    )

    parser.add_argument(
        "--horisontal-kernel-length",
        help="Kernel length for removing horisontal lines",
        type= int,
        default=None,
        required=False
    )

    parser.add_argument(
        "--blue-color-filter",
        nargs="+",
        help="Three band BGR color filter. Specify as a sequence of six integers,\
              e.g. (90, 10, 10, 255, 90, 90)",
        required=False,
        default=None
    )

    parser.add_argument(
        "--red-color-filter",
        nargs="+",
        help="Three band BGR color filter. Specify as a sequence of six integers,\
             e.g. (90, 10, 10, 255, 90, 90)",
        required=False,
        default=None
    )

    parser.add_argument(
        "--x-interval",
        help="The x-interval (height) in which to remove horisontal lines",
        nargs=2,
        type=int,
        required=False,
        default=None
    )

    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()

    if args.horisontal_kernel_length is not None and args.x_interval is None:
        raise argparse.ArgumentError(
            "Expecting -interval if horisontal-kernel-length is set"
        )

    run(
        input_image_path=args.input,
        output_directory=args.output,
        identifier=args.name,
        smooth_kernel_size=args.smooth_kernel_size,
        threshold_value=args.threshold_value,
        background_kernel_size=args.background_kernel_size,
        scale=args.scale,
        debug=args.debug,
        horisontal_kernel_length=args.horisontal_kernel_length,
        blue_color_filter=args.blue_color_filter,
        red_color_filter=args.red_color_filter,
        x_interval=args.x_interval
    )


if __name__ == "__main__":
    main()
