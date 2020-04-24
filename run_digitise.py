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


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input-directory",
        help="Input directory",
        type=Path,
        required=True,
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

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    pattern = re.compile("(\d+)\.png")
    for trace in args.input_directory.iterdir():
        if "annotated" in trace.stem or trace.suffix != ".png":
            continue
        pattern_matches = pattern.findall(str(trace))
        if len(pattern_matches) != 1:
            logger.info("Skipping trace, could not find trace number.")
        split_number = int(pattern_matches[0])
        logger.info(f"digitising {trace}")
        digitise_trace(
            trace,
            output_directory=args.input_directory,
            session_number=args.session_number,
            scale=args.scale
        )


if __name__ == "__main__":
    main()
