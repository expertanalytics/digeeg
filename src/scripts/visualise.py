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


def analyse(*, data_array: np.ndarray, sampling_frequency: int, out_file: Path = None) -> None:
    """Plot input arras and PSD of the input array.

    Arguments:
        data_array - Array of shape(N, 2). array[:, 0] is treated as time, array[:, 1] is the voltage.
    """
    fig, (axl, axr) = plt.subplots(ncols=2, figsize=(15, 10))

    time = data_array[:, 0]
    data = data_array[:, 1]

    axl.plot(time, data)
    axl.set_xlabel("Time")
    axl.set_ylabel("mV/cm")

    frequencies, power_density = welch(
        data,
        fs=sampling_frequency,
        nperseg=250,
        noverlap=None,          # ???
        return_onesided=True,   # ???
        detrend="constant"      # ???
    )

    axr.plot(frequencies, power_density)
    axr.set_xscale("log")
    axr.set_yscale("log")
    axr.set_xlabel("Frequency [Hz]")
    axr.set_ylabel("Power [dB]")

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


def concatenate_arrays(array_list: tp.Iterable[np.ndarray]) -> np.ndarray:
    return np.concatenate(array_list)


def create_parser():
    parser = argparse.ArgumentParser("Compute and plot the PSD of a sequence of itme series.")

    parser.add_argument(
        "-i",
        "--input",
        nargs="+",
        help="List of EEG traces stored as .npy",
        type=Path,
        required=True
    )

    parser.add_argument(
        "--output-image",
        help="Output file name",
        type=Path,
        required=False,
        default=None,
    )

    parser.add_argument(
        "--output-dataset",
        help="File path to hdf5 dataset.",
        type=Path,
        required=False,
        default=None
    )

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    list_of_data = handle_input_data(args.input)

    if args.output_dataset is not None:
        save_arrays(list_of_data, args.output_dataset)

    data_array = concatenate_arrays(list_of_data)
    analyse(data_array=data_array, sampling_frequency=140, out_file=args.output_image)


if __name__ == "__main__":
    main()
