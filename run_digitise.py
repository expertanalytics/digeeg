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
    trace_number: int,
    scale: float
) -> None:
    try:
        subprocess.check_output([
            "digitise-traces",
            "-i", input_file,
            "-o", str(output_directory),
            "-n", f"{trace_number}",
            "--scale", f"{scale}"
        ])
    except subprocess.CalledProcessError as e:
        logger.error(e)


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
        "--scale",
        help="Number of pixels per 15 cm.",
        type=float,
        required=False,
        default=None
    )

    return parser


def get_scale(args: argparse.Namespace) -> float:
    scale_directory = args.input_directory.parents[0]
    try:
        scale_path = (scale_directory / "scales.txt")
        if not scale_path.exists():
            logger.error(f"Could not find scale file. Tried to open {scale_path}. Specify scale manually")
            return None
        pattern = re.compile("(\d+)")
        target_id = int(pattern.findall(str(args.input_directory.parts[-1]))[0])
        with scale_path.open("r") as scale_file:
            for line in scale_file.readlines():
                case, scale = line.split(",")
                if int(case.split(":")[1]) == target_id:
                    return float(scale.split(":")[1])
    except IndexError:
        logger.error("Could not parse scale file. Specify scale manually.")
        return None
    return None


def main():
    parser = create_parser()
    args = parser.parse_args()

    # read scale
    if args.scale is None:
        scale = get_scale(args)
        if scale is None:
            raise ValueError("Please specify scale manualy")
    else:
        scale = args.scale

    pattern = re.compile("(\d+)\.png")
    for trace in args.input_directory.iterdir():
        if "annotated" in trace.stem or trace.suffix != ".png":
            continue
        pattern_matches = pattern.findall(str(trace))
        if len(pattern_matches) != 1:
            logger.info("Skipping trace, could not find trace number.")
        trace_number = int(pattern_matches[0])
        logger.info(f"digitising trace {trace_number}")
        digitise_trace(
            trace,
            output_directory=args.input_directory,
            session_number=trace_number,
            scale=scale
        )


if __name__ == "__main__":
    main()
