import numpy as np
import typing as tp
import matplotlib.pyplot as plt

import h5py
import argparse
import logging
import os
import re

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from scipy.signal import welch


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


from dgutils import (
    read_number_of_traces,
)


def plot_traces(*, data_array: np.ndarray, out_file: Path) -> None:
    fig, ax = plt.subplots(figsize=(20, 8))
    time = data_array[:, 0]
    data = data_array[:, 1]

    ax.plot(time, data)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("$\mu V$")

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
    start_time: float,
    stop_time: float,
    duration: float,
    voltage_scale: int = 200
) -> np.ndarray:
    """
    time is data_array[:, 0].
    time is data_array[:, 1].

    NB! Operates in place.

    Voltage scale is micro V / cm.,
    """
    data_array[:, 0] -= data_array[0, 0]    # -= t0
    time_scale = data_array[:, 0].max()/duration

    data_array[:, 0] /= time_scale     # time interval is now (0, time_duration)
    data_array[:, 0] += start_time     # time interval is not (start_time, stop_time)

    tolerance = 0.1     # How close the scaling should be to match stop-time
    if stop_time is not None:
        if abs(stop_time - data_array[-1, 0]) > tolerance:
            _data_array = np.zeros(shape=(data_array.shape[0] + 1, data_array.shape[1]))
            _data_array[:-1, :] = data_array
            _data_array[-1, ...] = [stop_time, 0]       # Place a zero to demarkate end of split
            data_array = _data_array

    # Apply voltage scale and mean center
    data_array[:, 1] -= data_array[:, 1].mean()
    data_array[:, 1] *= voltage_scale

    if flip_time:
        data_array[:, 1] = data_array[:, 1][::-1]
    if flip_voltage:
        data_array[:, 1] *= -1
    return data_array


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
    parser = argparse.ArgumentParser(
        """Join EEG traces together. Use max time to scale the duration
        of the time trace. Typically, 25 mm is equivalent to one second.

        Flip time and flip traces are used to flip the time and voltage axes. For "normal"-looking
        traces use 'flip-voltage'.
        """
    )
    parser.add_argument(
        "--upper",
        nargs="+",
        help="List of traces in the 'upper' EEG trace. Filenames are 'trace{d:}.npy'",
        type=int,
        required=False,
        default=None
    )

    parser.add_argument(
        "--lower",
        nargs="+",
        help="List of traces in the 'lower' EEG trace. Filenames are 'trace{d:}.npy'",
        type=int,
        required=False,
        default=None
    )

    parser.add_argument(
        "--output-directory",
        help="File path to npy dataset.",
        type=Path,
        required=False,
    )

    parser.add_argument(
        "--split-id",
        help="String to identify the split within the trace",
        type=int,
        required=True
    )

    parser.add_argument(
        "--flip-voltage",
        help="Multiply voltage by -1.",
        action="store_false",
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
        help="micro volts per cm. Defaults to 200.",
        required=False,
        default=200,
        type=int
    )

    parser.add_argument(
        "--start-time",
        help="Start time in seconds (follow the time stamp). There is typically 25 mm per second.",
        required=False,
        default=0,
        type=float
    )

    parser.add_argument(
        "--stop-time",
        help="Stop time in seconds. (Follow the time stamp). There is typically 25 mm per seconds.",
        required=False,
        default=0,
        type=float
    )

    parser.add_argument(
        "--duration",
        help="Stop time in seconds. (Follow the time stamp). There is typically 25 mm per seconds.",
        required=False,
        default=6,
        type=float
    )

    return parser


def _join_traces(
    traces: tp.List[int],
    output_directory: Path,
    eeg_name: str,
    split_id: int,
    flip_time: bool,
    flip_voltage: bool,
    start_time: float,
    stop_time: float,
    duration: float,
    voltage_scale: float,
) -> None:
    # Array of trace filenames to include
    filename_list = np.asarray([Path(f"trace{number}.npy") for number in traces])

    # Array of time series
    split_id_path = Path(f"split{split_id}")
    if Path.cwd() == split_id_path:
        logger.info("Looking for traces in current directory")
    else:
        if not split_id_path.is_dir():
            raise FileNotFoundError(f"Could not find directory: {split_id_path}")
        else:
            logger.info(f"Looking for traces in {split_id_path}")
            filename_list = [split_id_path / p for p in filename_list]
    list_of_arrays = np.asarray(handle_input_data(filename_list))

    # Concatenate the arrays
    data_array = concatenate_arrays(list_of_arrays)
    data_array = scale_arrays(
        data_array=data_array,
        flip_time=flip_time,
        flip_voltage=flip_voltage,
        start_time=start_time,
        stop_time=stop_time,
        duration=duration,
        voltage_scale=voltage_scale
    )

    # save_arrays(data_array, args.output_directory / f"eeg_{args.split_id}_{args.eeg_name}.h5")
    if output_directory is None:
        cwd_list = list(Path.cwd().parts)
        if Path.cwd() == split_id_path:
            _patient, _round, _session, _ = cwd_list[-4:]
        else:
            _patient, _round, _session = cwd_list[-3:]

        number_pattern = re.compile("(\d+)")
        try:
            patient_number = int(number_pattern.search(_patient).group(1))
            round_number = int(number_pattern.search(_round).group(1))
            session_number = int(number_pattern.search(_session).group(1))
        except (ValueError, AttributeError):
            msg = "Could not determine correct output directory, please set --output-directory manually'"
            logger.error(msg)
            raise

        output_directory = Path(f"timeseries_E{patient_number}_R{round_number}_S{session_number}")

    if not output_directory.is_dir():
        output_directory.mkdir(exist_ok=True)

    save_array_numpy(data_array, output_directory / f"eeg_{split_id}_{eeg_name}")     # appends .npy
    plot_traces(data_array=data_array,
        out_file=output_directory / f"eeg_{split_id}_{eeg_name}.png"
    )


def _validate_args(args: tp.Any) -> None:
    if args.start_time >= args.stop_time:
        raise ValueError("Start time ({args.start_time}) >= stop time ({args.stop_time}).")


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()

    logname = Path(f"log_{args.split_id}")
    with logname.open("w") as wfh:
        wfh.write(f"flip_time: {args.flip_time}\n")
        wfh.write(f"flip_voltage: {args.flip_voltage}\n")
        wfh.write(f"start_time: {args.start_time}\n")
        wfh.write(f"stop_time: {args.stop_time}\n")
        wfh.write(f"voltsage_scale: {args.voltage_scale}\n")
        wfh.write(f"output_directory: {args.output_directory}\n")
        wfh.write(f"split_id: {args.split_id}\n")
        wfh.write(f"upper: {args.upper}\n")
        wfh.write(f"lower: {args.lower}\n")

    upper = set()
    if args.upper is None:
        logger.info("Skipping upper eeg. No traces supplied")
    else:
        upper = set(args.upper)

    lower = set()
    if args.lower is None:
        logger.info("Skipping upper eeg. No traces supplied")
    else:
        lower = set(args.lower)

    intersection = upper & lower
    if intersection == set():
        if len(upper) > 0:
            _join_traces(
                args.upper,
                args.output_directory,
                "upper",
                args.split_id,
                args.flip_time,
                args.flip_voltage,
                args.start_time,
                args.stop_time,
                args.duration,
                args.voltage_scale
            )
        if len(lower) > 0:
            _join_traces(
                args.lower,
                args.output_directory,
                "lower",
                args.split_id,
                args.flip_time,
                args.flip_voltage,
                args.start_time,
                args.stop_time,
                args.duration,
                args.voltage_scale
            )
    else:
        raise ValueError(f"Traces {intersection} supplied to both upper and lower")
