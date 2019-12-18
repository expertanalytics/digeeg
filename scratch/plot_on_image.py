import numpy as np
import matplotlib.pyplot as plt

import cv2

from pathlib import Path

from dgimage import load_image


def read_line(id_number):
    in_array = np.load(f"tmp_lines/out_array{id_number}.npy")
    return in_array[:, 0], in_array[:, 1]


background = load_image(Path("scan4_tmp_splits/split4_scan4.pkl"))
background.image = background.image[575:, 60:7350]


M, N, *_ = background.image.shape
fig, ax = plt.subplots(1)
ax.imshow(background.image)

scale = 15/2693      # Number of pixels between two black dots
scale = 15/background.scale
# scale = 1

step = 300
offset_list = list(map(lambda x: x - 575, [2817, 2845, 2170, 1500, 1470, 1500, 824, 820]))
color_list = ["firebrick", "darkorange", "yellow", "chartreuse",
              "green", "steelblue", "slateblue", "magenta"]
for id_number in range(8):      # 8
    x, y = read_line(id_number + 1)
    color = color_list[id_number]

    ax.plot(x - 60, y*-1 + offset_list[id_number], linewidth=1, linestyle="-.", color=color)

ax.set_xticklabels([f"{tick*scale:.1f}cm" for tick in ax.get_xticks()])
ax.set_yticklabels([f"{tick*scale:.1f}cm" for tick in ax.get_yticks()])

plt.show()
