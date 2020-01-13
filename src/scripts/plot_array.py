import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import argparse
import logging
import os


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


def run(input_file: Path) -> None:
    """Reat and plot the input file.

    Plot x[:, 0] vs x[:, 1].
    """
    input_array = np.load(str(input_file))
    fig, ax = plt.subplots(1)
    ax.plot(input_array[:, 0], input_array[:, 1])
    plt.show()


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot a line")
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        help="Path to digitised trace as compressed numpy array",
        required=True
    )

    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    run(args.input)


if __name__ == "__main__":
    main()
