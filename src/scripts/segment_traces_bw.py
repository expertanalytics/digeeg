import typing as tp
import numpy as np
import matplotlib.pyplot as plt

import cv2
import itertools
import math
import argparse
import logging
import os

from pathlib import Path
from shapely.geometry import Polygon

from dgimage import (
    read_image,
    save_image,
    Image,
    mtableau_brg,
    color_to_256_BRG,
    color_to_256_RGB,
)

from dgutils import (
    get_debug_path,
    filter_contours,
    get_contours,
    match_contours,
    get_graph_matcher,
    save,
    plot,
)


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


def run(
    *,
    input_image_path: Path,
    output_directory: Path,
    identifier: str,
    scale: float = None,
    debug: bool = False,
):
    """Segment the contours from a black and white image and save the segmented lines."""
    image = read_image(input_image_path)
    if debug:
        debug_path = get_debug_path("remove_background")
        save(image.image, debug_path, "input")
    image.bgr_to_gray()

    if debug:
        save(image.iamge, debug_path, "match_contours")
    contours = get_contours(image=image)
    features = match_contours(matcher=get_graph_matcher(), contours=contours)

    if debug:
        debug_path = get_debug_path("extract_contours_bw")
        save(image.image, debug_path, "input")

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
            xy=(x0, y1),
            xycoords="data",
            xytext=(0, 35),
            textcoords="offset points",
            size=10,
            bbox=dict(
                boxstyle="round",
                fc=color       # normalised color
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
        "--scale",
        help="The scale of the image. What is the phusical size of a pixel.",
        default=None,
        type=int,
        required=False
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Store intermediate images for calibration.",
        required=False
    )
    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()

    run(
        input_image_path=args.input,
        output_directory=args.output,
        identifier=args.name,
        scale=args.scale,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
