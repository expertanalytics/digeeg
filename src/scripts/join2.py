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
    voltage_scale: int = 200
) -> tp.Tuple[np.ndarray, np.ndarray]:
    """
    Voltage scale is measured in micro volts per cm.
    """
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

    # Compute time scale
    time_scale = _time.max() / max_time
    _time *= 1/time_scale

    # TODO: Check this -- should I normalise
    _voltage -= _voltage.mean()
    _voltage *= 1/time_scale
    _voltage *= voltage_scale     # micro volts

    if flip_time:
        _time = _time[::-1]
    if flip_voltage:
        _voltage *= -1
    return _time, _voltage


def join_datasets(
    dataset_list: tp.List[tp.Tuple[np.ndarray, np.ndarray]]
) -> tp.Tuple[np.ndarray, np.ndarray]:
    time_list = []
    voltages_list = []

    time_list.append(dataset_list[0][0])
    voltages_list.append(dataset_list[0][1])
    last_voltage = voltages_list[-1][-1]
    last_time = time_list[-1][-1]
    for i in range(1, len(dataset_list)):
        time_list.append(dataset_list[i][0][:-140] + last_time)

        # TODO: Skal jeg legge til noe (last_voltage?) her?
        voltages_list.append(dataset_list[i][1][:-140])
        last_time = time_list[-1][-1]
        last_voltage = voltages_list[-1][-1]

    time_array = concatenate_arrays(time_list)
    voltage_array = concatenate_arrays(voltages_list)
    return time_array, voltage_array


def plot_entire_time_series(time, voltage, name):
    # TODO: set dynamic fig size
    fig, ax = plt.subplots(1, figsize=(30, 8), tight_layout=True)
    ax.plot(time, voltage)

    ax.set_xlabel("time: s")
    ax.set_ylabel("voltage $\mu V$")
    fig.savefig(f"{name}.png")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--eegs", help="Paths to eegs to join together. Expecting .h5-files.",
        type=Path,
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


def main():
    parser = create_parser()
    args = parser.parse_args()

    dataset_list = [
        read_dataset(
            filename,
            max_time=args.max_time,
            voltage_scale=args.voltage_scale
        ) for filename in args.eegs
    ]
    time, voltage = join_datasets(dataset_list)
    out_path = Path(f"{args.name}")
    # save_arrays(time, voltage, out_path)
    save_arrays_numpy(time, voltage, out_path)

    plot_entire_time_series(time, voltage, args.name)
    plt.show()


if __name__ == "__main__":
    main()
