from enum import Enum
import numpy as np

import matplotlib.colors as mcolors


class Colors(Enum):

    RED = (0, 0, 255)
    YELLOW = (0, 255, 255)
    LIME = (0, 255, 0)
    AQUA = (255, 255, 0)
    BLUE = (255, 0, 0)
    FUCHSIA = (255, 0, 255)

    MAROON = (0, 0, 125)
    OLLIVE = (0, 125, 125)
    GREEN = (0, 125, 0)
    TEAL = (125, 125, 0)
    NAVY = (125, 0, 0)
    PURPLE = (125, 0, 125)

    WHITE = (255, 255, 255)
    SILVER = (192, 192, 192)
    GAINSBORO = (220, 220, 220)
    GRAY = (125, 125, 125)
    BLACK = (0, 0, 0)

    @property
    def bgr(self):
        return self.value

    @property
    def bgr_normal(self):
        return tuple(map(lambda x: x/255, self.value))

    def dist(self, other) -> float:
        return np.linalg.norm(np.array(self.value) - other)


def mtableau_brg():
    return (mcolors.to_rgb(c) for c in mcolors.TABLEAU_COLORS.values())


def mbase_brg():
    return (mcolors.to_rgb(c) for c in mcolors.BASE_COLORS.values())


def mcss_brg():
    return (mcolors.to_rgb(c) for c in mcolors.CSS4_COLORS.values())


def color_to_256_BRG(color_tuple):
    return tuple(map(lambda x: x*255, color_tuple[::-1]))


def color_to_256_RGB(color_tuple):
    return tuple(map(lambda x: x*255, color_tuple))
