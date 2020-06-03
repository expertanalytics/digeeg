import scipy.signal as signal
from pathlib import Path

import matplotlib.pyplot as plt

import argparse

import h5py
import re
import numpy as np
import typing as tp


def concatenate_arrays(array_list: tp.Iterable[np.ndarray]) -> np.ndarray:
    return np.concatenate(array_list)


def save_arrays(time_array: np.ndarray, voltage_array: np.ndarray, out_path: Path) -> None:
    """Save list of arrays in as datasets with hdf5."""
    hdf5_file = h5py.File(f"{out_path}.h5", "w")
    hdf5_file.create_dataset("time", data=time_array)
    hdf5_file.create_dataset("voltage", data=voltage_array)
    hdf5_file.close()


def save_arrays_numpy(time: np.ndarray, voltage: np.ndarray, out_file: Path) -> None:
    array = np.zeros((time.size, 2))
    array[:, 0] = time
    array[:, 1] = voltage
    np.save(str(out_file), array)


def read_dataset(
    dataset_path: Path,
    flip_time: bool=False,
    flip_voltage: bool=False,
    max_time: float=6,
    voltage_scale: int = 1
) -> tp.Tuple[np.ndarray, np.ndarray]:
    """
    Voltage scale is measured in micro volts per cm.
    """

    pattern = re.compile("eeg_(\d+)_")
    split_number_matches = pattern.findall(str(dataset_path))

    if not len(split_number_matches) == 1:
        raise ValueError(f"could not parse file name {dataset_path}")
    split_number = int(split_number_matches[0])

    if dataset_path.suffix == ".h5":
        dts = h5py.File(dataset_path, "r")

        tlist = []
        ylist = []

        for key in dts.keys():
            t, y = dts[key]
            tlist.append(t)
            ylist.append(y)

        time_array = np.asarray(tlist)
        voltage_array = np.asarray(ylist)

        sorted_indices = np.argsort(time_array)
        _time = time_array[sorted_indices]
        _voltage = voltage_array[sorted_indices]
    elif dataset_path.suffix == ".npy":
        array = np.load(str(dataset_path))
        _time = array[:, 0]
        _voltage = array[:, 1]
    else:
        raise ValueError(
            f"Unknown file extension, got {dataset_path.suffix}, expecting '.npy' or '.h5'"
        )

    # Compute time scale
    time_scale = _time.max() / max_time
    _time *= 1/time_scale

    # TODO: Check this -- should I normalise
    _voltage -= _voltage.mean()
    _voltage *= 1/time_scale
    _voltage *= voltage_scale     # micro volts

    # Add multipe of max_time based on filename (split number)

    if flip_time:
        _voltgate = _voltage[::-1]      # time-axis stays the same. It is just an axis
    if flip_voltage:
        _voltage *= -1

    # Add max_time times split number to account for gaps in the data
    _time += split_number*max_time
    return _time, _voltage


def join_datasets(
    dataset_list: tp.List[tp.Tuple[np.ndarray, np.ndarray]]
) -> tp.Tuple[np.ndarray, np.ndarray]:
    time_list = []
    voltages_list = []

    time_list.append(dataset_list[0][0])
    voltages_list.append(dataset_list[0][1])
    last_time = time_list[-1][-1]
    for i in range(1, len(dataset_list)):
        # time_list.append(dataset_list[i][0][:-140] + last_time)
        time_list.append(dataset_list[i][0][:-140])
        voltages_list.append(dataset_list[i][1][:-140])
        last_time = time_list[-1][-1]

    time_array = concatenate_arrays(time_list)
    voltage_array = concatenate_arrays(voltages_list)
    return time_array, voltage_array


def plot_entire_time_series(time, voltage, name, eeg_flag):
    # TODO: set dynamic fig size
    fig, ax = plt.subplots(1, figsize=(50, 8), tight_layout=True)
    ax.plot(time, voltage)

    ax.set_xlabel("time: s")
    ax.set_ylabel("voltage $\mu V$")
    fig.savefig(f"{name}_{eeg_flag}.png")


def parse_filenames(id_list: tp.Iterable[int], eeg_flag: str) -> tp.List[Path]:
    """NB! only works in current working directory. EEG-flag is either alower or upper"""
    # Look for all files matching "eeg_{:d}_{eeg_flag}."
    p = Path(".")
    file_list = []
    for i in id_list:
        file_glob = list(filter(lambda x: x.suffix != ".png", p.glob(f"eeg_{i}_{eeg_flag}.*")))     # passed to multiple generators
        if len(file_glob) == 0:
            raise ValueError(f"No matching files found. Looking for 'eeg_{i}_{eeg_flag}.*'")
        if len(file_glob) > 1:
            raise ValueError(f"Multiple EEGs found, please move all but one. Found {file_glob}.")
        file_list.append(file_glob[0])
    suffixes = set(map(lambda x: x.suffix, file_list))
    if len(suffixes) != 1:
        raise UserWarning(
            f"Multiple suffixes found in the EEG file list. Is this right? Found {suffixes}"
        )
    return file_list


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--eegs", help="Paths to eegs to join together. Expecting .h5-files xor .npy-files.",
        type=int,
        nargs="+",
        required=True
    )

    parser.add_argument(
        "-n",
        "--name",
        help="Name of output file. Will create both .h5 and .png",
        required=True
    )

    parser.add_argument(
        "--voltage-scale",
        help="micro volts per cm.",
        required=False,
        default=1,
        type=int
    )

    parser.add_argument(
        "--max-time",
        help="seconds per 15 cm",
        required=False,
        default=6,
        type=int
    )

    parser.add_argument(
        "--lower",
        help="Look for eegs from the 'lower' trace.",
        action="store_true",
        required=False,
    )

    parser.add_argument(
        "--upper",
        help="Look for eegs from the 'upper' trace.",
        action="store_true",
        required=False,
    )

    parser.add_argument(
        "--flip-time",
        help="reverse the time-axis",
        action="store_true",
        required=False
    )

    parser.add_argument(
        "--flip-voltage",
        help="Invert the voltage",
        action="store_true",
        required=False
    )

    return parser


def _validate_arguments(arguments: tp.Any) -> None:
    if arguments.upper and arguments.lower:
        raise ValueError("only one of 'upper' or 'lower' can be set")
    if not arguments.upper and not arguments.lower:
        raise ValueError("One of 'upper' or 'lower' must be set")


def _get_flag(arguments: tp.Any) -> str:
    eeg_flag: str
    if arguments.upper:
        eeg_flag = "upper"
    elif arguments.lower:
        eeg_flag = "lower"
    return eeg_flag


def main():
    parser = create_parser()
    args = parser.parse_args()
    _validate_arguments(args)

    eeg_flag = _get_flag(args)
    filename_list = parse_filenames(args.eegs, eeg_flag)

    dataset_list = [
        read_dataset(
            filename,
            max_time=args.max_time,
            voltage_scale=args.voltage_scale,
            flip_time=args.flip_time,
            flip_voltage=args.flip_voltage,
        ) for filename in filename_list
    ]
    time, voltage = join_datasets(dataset_list)

    if args.upper:
        out_path = Path(f"{args.name}_upper")
    else:       # upper/lower exclusivity checked validate_arguments
        out_path = Path(f"{args.name}_lower")
    save_arrays_numpy(time, voltage, out_path)

    plot_entire_time_series(time, voltage, args.name, eeg_flag)
    plt.show()


if __name__ == "__main__":
    main()
