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


def split_image(
    input_file: Path,
    base_output_directory: Path,
    round_number: int,
    session_number: int
) -> Path:
    outpath = base_output_directory / f"round{round_number}_S{session_number}"
    try:
        subprocess.check_output([
            "split-image",
            "-i", str(input_file),
            "-o", outpath,
            "-n", f"session{session_number}",
            "--kernel-length", "0"
        ])
    except subprocess.CalledProcessError as e:
        print(e)
    return Path(outpath)


def segment_trace(
    input_file: Path,
    output_base_directory: Path,
    session_number: int,
    split_number: int,
    horisontal_lines: bool,
    color_filters: bool,
    scale: float
) -> Path:
    outpath = output_base_directory / f"split{split_number}"
    command = [
        "segment-traces",
        "-i", str(input_file),
        "-o", outpath,
        "-n", f"session{session_number}",
        "--scale", f"{scale}"
    ]

    if horisontal_lines:
        command += ["--horisontal-kernel-length", "500"]
        command += ["--x-interval", "3000", "-1"]

    if color_filters:
        command += [
            "--blue-color-filter", "90", "10", "10", "255", "90", "90",
            "--red-color-filter", "10", "10", "90", "90", "90", "255"
        ]

    try:
        subprocess.check_output(command)
    except subprocess.CalledProcessError as e:
        print(e)
    return Path(outpath)


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
        "--input",
        help="Input EEG",
        required=True,
    )

    parser.add_argument(
        "-o",
        "--output",
        help="Output base directory",
        required=True
    )

    parser.add_argument(
        "-id",
        "--patient-id",
        help="The patient identification number, without any 'E' prefix.",
        required=True,
    )

    parser.add_argument(
        "-r",
        "--round-number",
        help="The ECT round number. The first, second etc course of ECT.",
        required=True,
    )

    parser.add_argument(
        "-s",
        "--session-number",
        help="The session number within one ECT round or course.",
        required=True,
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
        "--clean-directory",
        help="Clean output directory. Use with caution",
        action="store_true",
        required=True,
    )

    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    # clean diectory

    output_base = Path(f"{args.output}{args.patient_id}")
    if args.clean_directory and output_base.is_dir():
        shutil.rmtree(output_base)
    output_base.mkdir(exist_ok=True, parents=True)

    split_directory = split_image(args.input, output_base, args.round_number, args.session_number)

    pattern = re.compile("(\d+)\.png")

    scale_path = split_directory / "scales.txt"
    scale_dict = {}
    with scale_path.open("r") as in_file:
        for line in in_file.readlines():
            case, scale = line.split(",")
            scale_dict[int(case.split(":")[1])] = float(scale.split(":")[1])

    # loop over all the splits and feed to segment_trace
    for split_child in split_directory.iterdir():
        logger.info(f"segmenting {split_child}")
        pattern_matches = pattern.findall(str(split_child))
        if len(pattern_matches) != 1:      # Wrong format to be a split
            continue
        split_number = int(pattern_matches[0])

        trace_directory = segment_trace(
            split_child,
            output_base / f"round{args.round_number}_S{args.session_number}",
            args.session_number,
            split_number,
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
