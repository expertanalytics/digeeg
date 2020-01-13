import numpy as np
import matplotlib.pyplot as plt

from IPython import embed


def read_line(id_number):
    in_array = np.load(f"tmp_lines/out_array{id_number}.npy")
    return in_array[:, 0], in_array[:, 1]


if __name__ == "__main__":
    step = 650
    offset_list = [0, 0, step, 2*step, 2*step, 2*step, 3*step, 3*step]
    lines = []
    fig, ax = plt.subplots(1)
    for id_number in range(8):
        x, y = read_line(id_number)
        ax.plot(x, y + offset_list[id_number], linewidth=1)
    ax.grid(True)

    scale = 1752
    ax.set_title("The digitised paper strip", fontsize=16)
    ax.set_xticklabels(["{:.1f} cm".format(i/scale) for i in ax.get_xticks()])
    ax.set_yticklabels(["{:.1f} cm".format(i/scale) for i in ax.get_yticks()])
    ax.set_ylabel("Voltage", fontsize=16)
    ax.set_xlabel("Time", fontsize=16)

    ax.legend([
        "ECG", "ECG",
        "EMG",
        "EEG2", "EEG2", "EEG2",
        "EEG1", "EEG1"
    ])

    fig.tight_layout()
    plt.show()
    # embed()
