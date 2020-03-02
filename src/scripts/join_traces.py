import numpy as np
import typing as tp
import matplotlib.pyplot as plt

import h5py
import argparse
import logging
import os

from pathlib import Path
from scipy.signal import welch


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


from dgutils import (
    sort_bounding_boxes,
    read_number_of_traces,
)


def analyse(*, data_array: np.ndarray, sampling_frequency: int, out_file: Path = None) -> None:
    """Plot input arras and PSD of the input array.

    Arguments:
        data_array - Array of shape(N, 2). array[:, 0] is treated as time, array[:, 1] is the voltage.
    """
    fig, (axu, axd) = plt.subplots(nrows=2, figsize=(15, 10))
    time = data_array[:, 0]
    data = data_array[:, 1]

    axu.plot(time, data)
    axu.set_xlabel("Time")
    axu.set_ylabel("mV/cm")

    frequencies, power_density = welch(
        data,
        fs=sampling_frequency,
        nperseg=250,
        noverlap=None,          # ???
        return_onesided=True,   # ???
        detrend="constant"      # ???
    )

    axd.plot(frequencies, power_density)
    axd.set_xscale("log")
    axd.set_yscale("log")
    axd.set_xlabel("Frequency [Hz]")
    axd.set_ylabel("Power [dB]")

    if out_file is not None:
        fig.savefig(out_file)

    plt.show()
    plt.close(fig)


def save_arrays(array_list: tp.Iterable[np.ndarray], out_file):
    """Save list of arrays in as datasets with hdf5."""
    hdf5_file = h5py.File(str(out_file), "w")
    for i, array in enumerate(array_list):
        hdf5_file.create_dataset(f"series{i}", data=array)
    hdf5_file.close()


def concatenate_arrays(array_list: tp.Iterable[np.ndarray]) -> np.ndarray:
    return np.concatenate(array_list)


def read_arrays(filename_list: tp.Iterable[Path]) -> tp.List[np.ndarray]:
    return list(map(np.load, filename_list))


def read_dataset(filename: Path) -> tp.List[np.ndarray]:
    """Return a list of datasets in a hdf4 file."""
    f = h5py.File(str(filename), "r")
    list_of_arrays = [np.asarray(f[key]) for key in f.keys()]
    f.close()
    return list_of_arrays


def handle_input_data(filename_list: tp.Iterable[Path]) -> tp.List[np.ndarray]:
    """Return a list of all input array, supplied eirher as hdf5 or .npy."""
    list_of_data: tp.List[np.ndarray] = []
    for fname in filename_list:
        if fname.suffix == ".hdf5":
            list_of_data += read_dataset(fname)
        elif fname.suffix == ".npy":
            list_of_data += read_arrays([fname])
        else:
            logging.info(f"unknown file extension {fname.suffix}")
    return list_of_data


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Join EEG traces together")
    parser.add_argument(
        "--exclude-traces",
        nargs="+",
        help="List of EEG trace ids to exclude. Filenames are 'trace{d:}.npy'",
        type=int,
        required=True
    )

    parser.add_argument(
        "--output-directory",
        help="File path to hdf5 dataset.",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--split-id",
        help="String to identify the split within the trace",
        type=int,
        required=True
    )

    parser.add_argument(
        "--eeg1",
        nargs="+",
        help="Id of a trace in EEG1",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--eeg2",
        nargs="+",
        help="Id of a trace in EEG2",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--flip",
        help="Multiply eeg trace by -1.",
        action="store_true",
        required=False
    )

    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()

    # Array of all trace ids
    all_trace_ids = np.asarray(read_number_of_traces(Path.cwd()))
    exclude_ids = np.asarray(args.exclude_traces)
    include_ids = np.setdiff1d(all_trace_ids, exclude_ids)

    # Array of trace filenames to include
    filename_list = np.asarray([Path(f"trace{number}.npy") for number in include_ids])

    # Array of time series
    array_of_data = np.asarray(handle_input_data(filename_list))

    # Indices of the two timeseries. Indices relate to `array_of_data` and `filename_list`
    s1, s2 = sort_bounding_boxes(array_of_data)
    if args.eeg1 in include_ids[s1]:
        assert args.eeg2 in include_ids[s2], "eeg1 in s1, BUT eeg2 id not in s2"
    elif args.eeg1 in include_ids[s2]:
        assert args.eeg2 in include_ids[s1], "eeg1 in s2, BUT eeg2 id not in s1"
        s1, s2 = s2, s1     # s1 is sequence 1

    print(filename_list[s1])
    print(filename_list[s2])

    s1_data_array = concatenate_arrays(array_of_data[s1])
    s2_data_array = concatenate_arrays(array_of_data[s2])

    if args.flip:
        s1_data_array[:, 1] *= -1
        s2_data_array[:, 1] *= -1

    print("s1: ", s1_data_array[:, 1].max())
    print("s2: ", s2_data_array[:, 1].max())

    # TODO: How to tell the difference between eeg1 and eeg2. Need to look at the y-values.
    save_arrays(s1_data_array, args.output_directory / f"{args.split_id}_eeg1.h5")
    save_arrays(s2_data_array, args.output_directory / f"{args.split_id}_eef2.h5")

    # NB! The images are stored in cwd
    analyse(data_array=s1_data_array, sampling_frequency=140, out_file=Path("eeg1.png"))
    analyse(data_array=s2_data_array, sampling_frequency=140, out_file=Path("eeg2.png"))
