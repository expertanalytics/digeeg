#!/usr/bin/env python

import typing as tp

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
    output_directory: Path,
    session_number: int,
    horisontal_lines: bool,
    color_filters: bool,
    scale: float,
    x_interval: tp.Tuple[int, int]
) -> Path:
    command = [
        "segment-traces",
        "-i", str(input_file),
        "-o", output_directory,
        "-n", f"session{session_number}",
        "--scale", f"{scale}"
    ]

    if horisontal_lines:
        command += ["--horisontal-kernel-length", "500"]
        command += ["--x-interval"]
        if x_interval is None:
            command += ["3000", "-1"]
        else:
            command += [str(x_interval[0]), str(x_interval[1])]

    if color_filters:
        command += [
            "--blue-color-filter", "90", "10", "10", "255", "90", "90",
            "--red-color-filter", "10", "10", "90", "90", "90", "255"
        ]

    try:
        subprocess.check_output(command)
    except subprocess.CalledProcessError as e:
        logger.info(e)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        help="Input split image",
        required=True,
    )

    parser.add_argument(
        "-o",
        "--output",
        help="Output base directory",
        required=True
    )

    parser.add_argument(
        "-s",
        "--session-number",
        help="The session number within one ECT round or course.",
        required=True,
    )

    parser.add_argument(
        "--scale",
        help="Number of pixels per 15 cm.",
        type=float,
        required=True
    )

    parser.add_argument(
        "--remove-horisontal-lines",
        help="Remove horisontal lines.",
        action="store_true",
        required=False
    )

    parser.add_argument(
        "--color-filter",
        help="Turn on blue and red color filters.",
        action="store_true",
        required=False
    )

    parser.add_argument(
        "--x-interval",
        help="Height in which to remove horisontal lines.",
        type=int,
        nargs="+",
        required=False,
        default=None,
    )

    return parser


def check_args(args: argparse.Namespace) -> None:
    """Check the consistency of the arguments."""
    if args.x_interval is not None and not args.remove_horisontal_lines:
        raise ValueError("Expected '--remove-horisontal-lines' if '--x-interval' is set.")


def main():
    parser = create_parser()
    args = parser.parse_args()
    check_args(args)

    segment_trace(
        args.input,
        args.output,
        args.session_number,
        args.remove_horisontal_lines,
        args.color_filter,
        args.scale,
        args.x_interval
    )


if __name__ == "__main__":
    main()
