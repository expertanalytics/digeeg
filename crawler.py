from pathlib import Path

import argparse
import re

import typing as tp
import numpy as np

import logging
import os

from collections import namedtuple


EEGTrace = namedtuple("EEGTrace", ["upper", "lower", "identifier"])


logging.basicConfig(level=os.environ.get("LOGLEVEL", "WARNING"))
logger = logging.getLogger(__name__)


def parse_directory(root_directory) -> tp.Generator[tp.Tuple[Path, int], None, None]:
    first_integer_pattern = re.compile(r"(\d+)")
    for branch_path in filter(lambda x: x.is_dir(), root_directory.iterdir()):
        logger.info(branch_path)

        matches = re.findall(first_integer_pattern, str(branch_path.stem))
        if len(matches) == 0:
            logger.warning(f"Could not parse {branch_path.resolve()}")
            continue
        logger.info(matches[0])
        yield branch_path.resolve(), int(matches[0])


def check_timeseries(
        session_dir,
        patient_number,
        round_number,
        session_number
) -> tp.Optional[EEGTrace]:
    """Return path to timeseires, and identifier number"""
    timeseires_dir = session_dir / "timeseries"
    if not timeseires_dir.is_dir():
        logger.warning(f"Could not find timeseries_dir {timeseires_dir.resolve()}")
        return None

    upper_pattern = re.compile(r"[Ee](\d+)[Rr](\d+)[Ss](\d+).*upper\.npy")
    lower_pattern = re.compile(r"[Ee](\d+)[Rr](\d+)[Ss](\d+).*lower\.npy")
    upper = None
    lower = None
    upper_identifiers = [None]
    lower_identifiers = [None]

    for trace_file in filter(lambda x: x.suffix == ".npy", timeseires_dir.iterdir()):
        # Jeg har en følelse av at dette ikke gikk bra

        upper_match = re.match(upper_pattern, str(trace_file.parts[-1]))
        lower_match = re.match(lower_pattern, str(trace_file.parts[-1]))

        if upper_match is not None:
            upper_identifiers = upper_match.groups()
            upper = trace_file.resolve()
        elif lower_match is not None:
            lower_identifiers = lower_match.groups()
            lower = trace_file.resolve()

    directory_identifiers = (patient_number, round_number, session_number)
    if not all(x in upper_identifiers for x in lower_identifiers):
        logger.warning(
            f"upper identifiers {upper_identifiers} does not match lower identifiers {lower_identifiers}"
        )
    elif not all(x in directory_identifiers for x in lower_identifiers):
        logger.warning(
            f"directory structure identifiers {directory_identifiers} do not match file identifiers {lower_identifiers}"
        )

    eeg_trace = EEGTrace(
        upper,
        lower,
        identifier=f"E{patient_number}R{round_number}S{session_number}"
    )
    return eeg_trace


def eeg_crawler(
        basedir: Path,
        patients: tp.Optional[tp.List[int]],
        rounds: tp.Optional[tp.List[int]],
        sessions: tp.Optional[tp.List[int]]
) -> tp.Generator[EEGTrace, None, None]:
    patient_list = list(parse_directory(basedir))
    for patients_directory, patient_number in patient_list:
        if patients is not None and patient_number not in patients:
            continue
        logger.info(patients_directory)
        round_list = list(parse_directory(patients_directory))
        if len(round_list) == 0:
            logger.warning(f"Could not parse any rounds in {patients_directory.resolve()}")
            continue

        for round_directory, round_number in round_list:
            if rounds is not None and round_number not in rounds:
                continue
            session_list = list(parse_directory(round_directory))
            if len(session_list) == 0:
                logger.warning(f"Could not parse any sessions in {round_directory.resolve()}")
                continue

            for session_directory, session_number in session_list:
                if sessions is not None and session_number not in sessions:
                    continue

                eeg_trace = check_timeseries(
                    session_directory,
                    patient_number,
                    round_number,
                    session_number
                )
                yield eeg_trace


def write_output(args, eeg_list: tp.List[EEGTrace], output_file: Path) -> None:
    with output_file.open("w") as ofile:
        ofile.write(f"basedirectory: {args.basedirectory}\n")
        ofile.write(f"patients: {args.patients}\n")
        ofile.write(f"rounds: {args.rounds}\n")
        ofile.write(f"sessions: {args.sessions}\n")
        ofile.write(f"output_file: {args.output_file}\n")
        ofile.write("\n")

        for eeg in eeg_list:
            ofile.write(f"{eeg.identifier}, upper, {eeg.upper}\n")
            ofile.write(f"{eeg.identifier}, lower, {eeg.lower}\n")

def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "basedirectory",
        type=Path,
        help="The root of the EEG folder hierarchy"
    )

    parser.add_argument(
        "-e",
        "--patients",
        type=int,
        nargs="+",
        help="Only include the specified patients",
        required=False
    )

    parser.add_argument(
        "-r",
        "--rounds",
        type=int,
        nargs="+",
        help="Only include the specified rounds",
        required=False
    )

    parser.add_argument(
        "-s",
        "--sessions",
        type=int,
        nargs="+",
        help="Only include the specified sessions",
        required=False
    )

    parser.add_argument(
        "-o",
        "--output-file",
        type=Path,
        help="where to store the filepaths"
    )

    return parser


def main() -> None:
    parser = create_argument_parser()
    args = parser.parse_args()
    eeg_list = list(eeg_crawler(args.basedirectory, args.patients, args.rounds, args.sessions))

    write_output(args, eeg_list, args.output_file)

    for eeg in eeg_list:
        print(eeg.identifier)
        if eeg.upper is not None:
            upper = np.load(eeg.upper)
        else:
            logger.warning(f"missing upper {eeg.identifier}")
        if eeg.lower is not None:
            lower = np.load(eeg.lower)
        else:
            logger.warning(f"missing upper {eeg.identifier}")

if __name__ == "__main__":
    main()
