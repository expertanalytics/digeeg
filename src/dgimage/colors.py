from enum import Enum
import numpy as np


class colors(Enum):

    BLUE = (255, 0, 0)
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)

    @property
    def bgr(self):
        return self.value

    def dist(self, other) -> float:
        return np.linalg.norm(np.array(self.value) - other)
