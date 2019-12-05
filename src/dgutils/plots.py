import numpy as np
import matplotlib.pyplot as plt

import cv2


def show(image_array: np.ndarray, mode: int = None):
    """Draw the current or supplied image."""
    if mode == "normal":
        cv2.namedWindow("ImageShow", cv2.WINDOW_NORMAL)

    # cv2.namedWindow("ImageShow", cv2.WINDOW_NORMAL)
    cv2.imshow("ImageShow", image_array)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def plot(image_array: np.ndarray, cmap: str = "gray") -> None:
    """Plot the current image."""
    fig, ax = plt.subplots(1)
    ax.imshow(image_array, cmap)
    plt.show()
    plt.close(fig)
