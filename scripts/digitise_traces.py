import numpy as np
import typing as tp
import matplotlib.pyplot as plt
import scipy.signal as signal

import os
import logging
import argparse

from pathlib import Path

from dgimage import read_image

from dgutils import PointAccumulator


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


logger = logging.getLogger(__name__)


def find_datapoints(
    *,
    image: np.ndarray,
    start_column: int,
    num_lines: int = 1,
    show: bool = False
) -> tp.Generator[tp.Tuple[int, float], None, None]:
    """Extract a time series from a segmented line or lines.

    Iterate through the image column by column. The column is convolved with a gaussian kernel and the
    peak is used as the "value".
    """
    _image = image
    window1 = signal.gaussian(50, 15)       # TODO: Expose parameters
    window1_sum = window1.sum()

    differentiator = PointAccumulator(num_lines=1)      # Restrict to a single line for now

    for i in range(start_column, _image.shape[1]):
        raw_signal = _image[:, i]
        filtered_signal = signal.fftconvolve(raw_signal, window1, mode='same')/window1_sum

        peaks = np.sort(signal.find_peaks(
            filtered_signal,
            prominence=5,
            distance=100
        )[0])       # sort so I can pick the "biggest" peak

        # Skip column if there are no peaks
        if len(peaks) == 0:
            continue

        # TODO: Check extrapolator
        new_points = differentiator.add_point(i, peaks, look_back=3)
        logger.debug(new_points)

        # Probably want to move away from generator. Use differentiator always
        # TODO: Why move away from generator?
        yield i, new_points[0]      # TODO: How are the poiints sorted?

        # TODO: These images should be stored somewhere for latrer quality review.
        if show:
            fig, (ax1, ax2) = plt.subplots(2)
            ax2.imshow(_image, cmap="gray")
            ax2.axvline(i, color="r")
            ax1.plot(raw_signal)
            ax1.plot(filtered_signal, "--")
            ax1.plot(peaks, filtered_signal[peaks], "x", linewidth=20)
            plt.show()
            plt.close(fig)


def run(
    input_image_path: Path,
    output_directory: Path,
    identifier: str,
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
        num_lines=0,
        show=show_step
    ):
        y_list.append(new_y)
        x_list.append(i)
        new_image[int(new_y), i] = 255      # For quality control

    x_arr = np.asarray(x_list, dtype=np.float_)
    y_arr = np.asarray(y_list, dtype=np.float_)

    y_arr -= y_arr.mean()      # mean zero
    y_arr *= -1                # flip

    # TODO: Save image for later review
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.imshow(new_image, cmap="gray")
    ax2.plot(x_arr, y_arr)
    fig.savefig(output_directory / f"{identifier}_image.png")
    if show:
        plt.show()
        plt.close(fig)

    out_array = np.zeros((x_arr.size, 2))
    out_array[:, 0] = x_arr
    out_array[:, 1] = y_arr

    output_directory.mkdir(exist_ok=True, parents=True)
    np.save(output_directory / f"{identifier}_trace", out_array)


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
        "--show_step",
        help="Display peak finding images for quality control",
        action="store_true",
        required=False,
    )

    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    run(args.input, args.output, args.name, args.start, args.show, args.show_step)
