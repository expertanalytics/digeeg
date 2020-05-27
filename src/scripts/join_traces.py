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
    read_number_of_traces,
)


def plot_traces(*, data_array: np.ndarray, out_file: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    time = data_array[:, 0]
    data = data_array[:, 1]

    ax.plot(time, data)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("mV/cm")

    if out_file is not None:
        fig.savefig(str(out_file))
    plt.close(fig)


def save_arrays(array_list: tp.Iterable[np.ndarray], out_file) -> None:
    """Save list of arrays in as datasets with hdf5."""
    hdf5_file = h5py.File(str(out_file), "w")
    for i, array in enumerate(array_list):
        hdf5_file.create_dataset(f"series{i}", data=array)
    hdf5_file.close()


def save_array_numpy(array: np.ndarray, out_file: Path) -> None:
    np.save(str(out_file), array)


def concatenate_arrays(array_list: tp.Iterable[np.ndarray]) -> np.ndarray:
    return np.concatenate(array_list)


def read_arrays(filename_list: tp.Iterable[Path]) -> tp.List[np.ndarray]:
    return list(map(np.load, filename_list))


def scale_arrays(
    data_array: np.ndarray,
    flip_time: bool,
    flip_voltage: bool,
    max_time:float=6,
    voltage_scale: int = 200
) -> np.ndarray:
    """
    time is data_array[:, 0].
    time is data_array[:, 1].

    NB! Operates in place.

    Voltage scale is micro V / cm.,
    """
    time_scale = data_array[:, 0].max()/max_time
    data_array /= time_scale

    data_array -= data_array[:, 1].mean()
    data_array[:, 1] *= voltage_scale

    if flip_time:
        data_array[:, 1] = data_array[:, 1][::-1]
    if flip_voltage:
        data_array[:, 1] *= -1


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


class CheckNameAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if values not in {"upper", "lower"}:
            msg = "Invalid EEG name. Expecting 'upper' or 'lower'"
            raise ValueError(msg)
        setattr(namespace, self.dest, values)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Join EEG traces together")
    parser.add_argument(
        "--traces",
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
        "-n",
        "--eeg-name",
        help="EEG name, either 'upper' or 'lower'",
        type=str,
        required=True,
        action=CheckNameAction
    )

    parser.add_argument(
        "--flip-voltage",
        help="Multiply voltage by -1.",
        action="store_true",
        required=False
    )

    parser.add_argument(
        "--flip-time",
        help="reverse the time-axis",
        action="store_true",
        required=False
    )

    parser.add_argument(
        "--voltage-scale",
        help="micro volts per cm.",
        required=False,
        default=200,
        type=int
    )

    parser.add_argument(
        "--max-time",
        help="seconds per 15 cm",
        required=False,
        default=6,
        type=int
    )

    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()

    logname = Path(f"log_{args.split_id}{args.eeg_name}")
    with logname.open("w") as wfh:
        wfh.write(f"eeg_name: {args.eeg_name}\n")
        wfh.write(f"flip_time: {args.flip_time}\n")
        wfh.write(f"flip_voltage: {args.flip_voltage}\n")
        wfh.write(f"max_time: {args.max_time}\n")
        wfh.write(f"voltsage_scale: {args.voltage_scale}\n")
        wfh.write(f"output_directory: {args.output_directory}\n")
        wfh.write(f"split_id: {args.split_id}\n")
        wfh.write(f"traces: {args.traces}\n")

    # Array of trace filenames to include
    filename_list = np.asarray([Path(f"trace{number}.npy") for number in args.traces])

    # Array of time series
    list_of_arrays = np.asarray(handle_input_data(filename_list))

    # Concatenate the arrays
    data_array = concatenate_arrays(list_of_arrays)
    scale_arrays(
        data_array=data_array,
        flip_time=args.flip_time,
        flip_voltage=args.flip_voltage,
        max_time=args.max_time,
        voltage_scale=args.voltage_scale
    )

    # save_arrays(data_array, args.output_directory / f"eeg_{args.split_id}_{args.eeg_name}.h5")
    save_array_numpy(data_array, args.output_directory / f"eeg_{args.split_id}_{args.eeg_name}")     # appends .npy
    plot_traces(data_array=data_array,
        out_file=args.output_directory / f"eeg_{args.split_id}_{args.eeg_name}.png"
    )
