#!/usr/bin/env python

import subprocess
import argparse
import re
import shutil

from pathlib import Path

# E ($2) -- Patient ID
# R ($3) -- Round 1, round 2 etc.
# S ($4) -- Session number


def split_image(
    input_file: Path,
    base_output_directory: Path,
    round_number: int,
    session_number: int
) -> Path:
    outpath = base_output_directory / f"round{round_number}_S{session_number}"
    subprocess.run([
        "split-image",
        "-i", str(input_file),
        "-o", outpath,
        "-n", f"session{session_number}"
    ])
    return Path(outpath)


def segment_trace(
    input_file: Path,
    output_base_directory: Path,
    session_number: int,
    split_number: int
) -> Path:
    outpath = output_base_directory / f"split{split_number}"
    subprocess.run([
        "segment-traces",
        "-i", str(input_file),
        "-o", outpath,
        "-n", f"session{session_number}"
    ])
    return Path(outpath)


def digitise_trace(
    input_file: Path,
    output_directory: Path,
    session_number: int,
) -> None:
    subprocess.run([
        "digitise-traces",
        "-i", input_file,
        "-o", str(output_directory),
        "-n", f"session{session_number}"
    ])


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        help="Input EEG",
        required=True,
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

    output_base = Path(f"splits_E{args.patient_id}")
    if args.clean_directory:
        shutil.rmtree(output_base)

    split_directory = split_image(args.input, output_base, args.round_number, args.session_number)

    pattern = re.compile("split(\d+)\.png")

    # loop over all the splits and feed to segment_trace
    for split_child in split_directory.iterdir():
        print(f"segmenting {split_child}")
        pattern_matches = pattern.findall(str(split_child))
        if len(pattern_matches) != 1:      # Wrong format to be a split
            continue

        split_number = int(pattern_matches[0])
        trace_directory = segment_trace(
            split_child,
            output_base / f"round{args.round_number}_S{args.session_number}",
            args.session_number,
            split_number
        )

        for trace_child in trace_directory.iterdir():
            if "annotated" in trace_child.stem:
                continue
            print(f"digitising {trace_child}")
            # Assume all trace_children are segented lines
            digitise_trace(trace_child, split_directory, args.session_number)


if __name__ == "__main__":
    main()
