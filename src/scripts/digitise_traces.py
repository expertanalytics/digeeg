import numpy as np
import typing as tp
import matplotlib.pyplot as plt

import os
import logging
import argparse

from pathlib import Path
from dgimage import read_image

import warnings
warnings.filterwarnings("ignore")


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

logger = logging.getLogger(__name__)


def find_datapoints(
    *,
    image: np.ndarray,
    start_column: int,
    show: bool = False
) -> tp.Iterator[tp.Tuple[int, float]]:
    """Extract a time series from a segmented line or lines.

    Iterate through the image column by column. The column is convolved with a gaussian kernel and the
    peak is used as the "value".
    """
    _image = image
    assert len(image.shape) == 2, "Expecting 2d image"

    for i in range(start_column, _image.shape[1]):
        raw_signal = _image[:, i]
        filtered_signal = raw_signal

        nonzero_indices = np.where(filtered_signal != 0)[0]

        # Check for a blank column
        if nonzero_indices.size == 0:
            continue

        median_nonzero_index = np.argsort(
            nonzero_indices
        )[nonzero_indices.size//2]
        median_index = nonzero_indices[median_nonzero_index]
        yield i, median_index

        # TODO: These images should be stored somewhere for latrer quality review.
        if show:
            fig, (ax1, ax2) = plt.subplots(2)
            ax2.imshow(_image, cmap="gray")
            ax2.axvline(i, color="r")
            ax1.plot(raw_signal)
            ax1.plot(filtered_signal, "--")
            ax1.plot(median_index, filtered_signal[median_index], "x", linewidth=20)
            plt.show()
            plt.close(fig)


def run(
    input_image_path: Path,
    output_directory: Path,
    identifier: str,
    scale: float = None,
    start_column: int = 0,
    show: bool = False,
    show_step: bool = False
):
    """Digitise *a single* eeg trace.

    input_image_path:
        Path to segmented EEG trace.
    output_directory:
        Path to directory where time series is stored.
    identifier:
        'Name' of the trace.
    start_column:
        The column from which to start digitising.
    show:
        Display peaks selection for each column and the final result.
    """
    trace_image = read_image(input_image_path)

    # Make sure image is grayscale
    if len(trace_image.image.shape) > 2:
        trace_image.bgr_to_gray()
    trace_image = trace_image.image

    new_image = np.zeros(trace_image.shape)

    x_list: tp.List[int] = []
    y_list: tp.List[float] = []

    for i, new_y in find_datapoints(
        image=trace_image,
        start_column=start_column,
        show=show_step
    ):
        y_list.append(new_y)
        x_list.append(i)
        try:
            new_image[int(new_y), i] = 255      # For quality control
        except IndexError as e:
            logging.info(f"{new_y}, {i}")
            logging.info(f"{new_image.shape}")
            import sys; sys.exit(1)

    x_arr = np.asarray(x_list, dtype=np.float_)
    y_arr = np.asarray(y_list, dtype=np.float_)
    y_arr -= y_arr.mean()      # mean zero

    # TODO: Save image for later review
    fig, (ax1, ax2) = plt.subplots(2)
    ax2.plot(x_arr, y_arr)
    output_directory.mkdir(exist_ok=True, parents=True)
    # fig.savefig(output_directory / f"trace{identifier}_imageQC.png")
    if show:
        plt.show()
        plt.close(fig)

    out_array = np.zeros((x_arr.size, 2))
    out_array[:, 0] = x_arr
    out_array[:, 1] = y_arr

    if scale is not None:
        out_array *= 15/scale

    output_directory.mkdir(exist_ok=True, parents=True)
    np.save(output_directory / f"trace{identifier}", out_array)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reduce a segmented trace to a line")
    parser.add_argument(
        "-i",
        "--input",
        help="Path to segmented image",
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
        help="Identifier of the EEG trace",
        type=str,
        required=True
    )

    parser.add_argument(
        "--scale",
        help="Scale the digitised trace. The actual scale is `15/scale`, or number of pixels in 15 cm",
        required=False,
        default=None,
        type=float,
    )

    parser.add_argument(
        "--start",
        help="Start to digitise from `start`",
        type=int,
        required=False,
        default=0
    )

    parser.add_argument(
        "--show",
        help="Display image of final line for quality control",
        action="store_true",
        required=False,
    )

    parser.add_argument(
        "--show-step",
        help="Display peak finding images for quality control",
        action="store_true",
        required=False,
    )

    return parser


def main() -> None:
    """Entrypoint."""
    parser = create_parser()
    args = parser.parse_args()
    run(args.input, args.output, args.name, args.scale, args.start, args.show, args.show_step)


if __name__ == "__main__":
    main()
