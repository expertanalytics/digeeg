from dataclasses import dataclass


@dataclass
class Line:
    """Oriented Lines in R^2. """

    a: float
    b: float
    c: float

    def __post_init__(self):
        if self.a == self.b == 0:
            raise ValueError("Parameters 'a' and 'b' cannot both be zero.")

        # Normalize -- Two lines are the same if parameters are the same up to
        # a _positive_ scaling factor, left side is positive

        norm = np.linalg.norm(dataclasses.astuple(self)[:2])
        self.a = self.a / norm
        self.b = self.b / norm
        self.c = self.c / norm

    def __call__(self, point) -> float:
        x, y = point
        return self.a * x + self.b * y + self.c

    def __neg__(self) -> "Line":
        return self.__class__(-self.a, -self.b, -self.c)

    def parallel_line(self, point) -> "Line":
        """Returns the parallel line passing through (x, y). """

        x, y = point
        a, b, c = dataclasses.astuple(self)

        # Orientation consistent with cross-product with pos. z-axis
        return self.__class__(a, b, -a * x - b * y)

    def orthogonal_line(self, point) -> "Line":
        """Returns the orthogonal line passing through (x, y). """

        x, y = point
        a, b, c = dataclasses.astuple(self)

        # Orientation consistent with cross-product with pos. z-axis
        return self.__class__(-b, a, -a * y + b * x)

    def project_point(self, point):
        orth = self.orthogonal_line(point)
        return self ^ orth

    def get_line_segment(self, image):
        """ Returns the line segment that intersects the image. """

        # TODO: Fix for when line more closely aligns with y-axis
        m, n = image.shape[:2]
        a, b, c = dataclasses.astuple(self)
        x0, y0 = (0, int(-c/b))
        x1, y1 = (n, int(-(a*n + c)/b))
        return ((x0, y0), (x1, y1))

    def __add__(self, other):
        a, b, c = dataclasses.astuple(self)
        return self.__class__(a, b, c + float(other))

    def __sub__(self, other):
        a, b, c = dataclasses.astuple(self)
        return self.__class__(a, b, c - float(other))

    def __ge__(self, point):
        return self(point) <= 0

    def __le__(self, point):
        return self(point) >= 0

    def __gt__(self, point):
        return self(point) < 0

    def __lt__(self, point):
        return self(point) > 0

    def __xor__(self, other):
        # Intersect
        A = np.vstack((dataclasses.astuple(self),
                       dataclasses.astuple(other)))

        x = np.linalg.solve(A[:, :2], -A[:, 2])
        return x
