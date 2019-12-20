import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import logging
import os


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


def run(input_array):
    fig, ax = plt.subplots(1)
    ax.plot(input_array[:, 0], input_array[:, 1])
    plt.show()


if __name__ == "__main__":
    import sys
    input_array = np.load(sys.argv[1])
    run(input_array)
