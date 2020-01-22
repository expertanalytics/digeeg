#!/usr/bin/env python

import subprocess
import argparse
import re
import shutil
import logging
import os
import sys

from pathlib import Path


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


def segment_trace(
    input_file: Path,
    output_path: Path,
    session_number: int,
    horisontal_lines: int,
    color_filters: bool,
    scale: float
) -> Path:
    command = [
        "segment-traces",
        "-i", str(input_file),
        "-o", str(output_path),
        "-n", f"session{session_number}",
        "--scale", f"{scale}"
    ]

    if horisontal_lines is not None:
        command += ["--horisontal-kernel-length", str(horisontal_lines)]

    if color_filters:
        command += ["--blue-color-filter", "--red-color-filter"]

    try:
        subprocess.check_output(command)
    except subprocess.CalledProcessError as e:
        print(e)
        sys.exit(1)
    return Path(output_path)


def digitise_trace(
    input_file: Path,
    output_directory: Path,
    session_number: int,
    scale: float
) -> None:
    try:
        subprocess.check_output([
            "digitise-traces",
            "-i", input_file,
            "-o", str(output_directory),
            "-n", f"{session_number}",
            "--scale", f"{scale}"
        ])
    except subprocess.CalledProcessError as e:
        print(e)
        sys.exit(1)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        help="Input EEG split",
        required=True,
    )

    parser.add_argument(
        "-o",
        "--output",
        help="Output base directory",
        required=True
    )

    parser.add_argument(
        "--remove-horisontal-lines",
        help="Remove horisontal lines.",
        type=int,
        default=None,
        required=False
    )

    parser.add_argument(
        "-s",
        "--session_number",
        help="The session number is used for naming output files.",
        required=True,
        type=int
    )

    parser.add_argument(
        "--color-filter",
        help="Turn on blue and red color filters.",
        action="store_true",
        required=False
    )

    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    # clean diectory

    scale_path = Path(args.input).parents[0] / "scales.txt"
    scale_dict = {}
    with scale_path.open("r") as in_file:
        for line in in_file.readlines():
            case, scale = line.split(",")
            scale_dict[int(case.split(":")[1])] = float(scale.split(":")[1])

    pattern = re.compile("(\d+)\.png")
    split_number = int(pattern.findall(str(args.input))[0])

    trace_directory = segment_trace(
        args.input,
        args.output,
        args.session_number,
        args.remove_horisontal_lines,
        args.color_filter,
        scale_dict[split_number]
    )

    for trace_child in trace_directory.iterdir():
        if "annotated" in trace_child.stem:
            continue
        trace_number = pattern.findall(str(trace_child))[0]
        logger.info(f"Digitising {trace_child}")
        # Assume all trace_children are segented lines
        # from IPython import embed; embed()
        # assert False, trace_number
        digitise_trace(trace_child, trace_child.parents[0], trace_number, scale_dict[split_number])


if __name__ == "__main__":
    main()
