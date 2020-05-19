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
    hdf5_file = h5py.File(str(out_path), "w")
    hdf5_file.create_dataset("time", data=time_array)
    hdf5_file.create_dataset("voltage", data=voltage_array)
    hdf5_file.close()


def save_arrays_numpy(time: np.ndarray, voltage: np.ndarray, out_file: Path) -> None:
    array = np.zeros((time.size, 2))
    array[:, 0] = time
    array[:, 1] = voltage
    np.save(str(out_file), array)


def _sort_key(key: str) -> int:
    pattern = "(\d+)"
    match = re.search(pattern, key)
    return int(match.group())


def read_dataset(dataset_path: Path, flip: bool=False) -> tp.Tuple[np.ndarray, np.ndarray]:
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

        if flip:
            voltage_array *= -1

        sorted_indices = np.argsort(time_array)
        return time_array[sorted_indices], voltage_array[sorted_indices]
    elif dataset_path.suffix == ".npy":
        array = np.load(str(dataset_path))
        return array[:, 0], array[:, 1]


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
        voltages_list.append(dataset_list[i][1][:-140])
        last_time = time_list[-1][-1]
        last_voltage = voltages_list[-1][-1]

    time_array = concatenate_arrays(time_list)
    voltage_array = concatenate_arrays(voltages_list)
    return time_array, voltage_array


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
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    dataset_list = [read_dataset(Path(filename)) for filename in args.eegs]
    time, voltage = join_datasets(dataset_list)
    out_path = Path(f"{args.name}.h5")
    # save_arrays(time, voltage, out_path)
    save_arrays_numpy(time, voltage, out_path)

    fig, ax = plt.subplots(1, figsize=(20, 10))
    ax.plot(time, voltage)
    fig.savefig(str(f"{args.name}.png"))
    plt.show()


if __name__ == "__main__":
    main()
