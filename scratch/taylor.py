import numpy as np
import typing as tp
import scipy.interpolate as interp

import matplotlib.pyplot as plt





class PointAccumulator:
    def __init__(self, num_lines):
        self.num_lines = num_lines
        self.list_of_values: tp.List[tp.List[float]] = [list() for i in range(num_lines)]

        self.x_points: tp.List[float] = []

    def add_point(self, x_val: int, point: np.ndarray, look_back: int = 4):
        _point = np.asarray(point)

        if len(self.x_points) >= look_back:      # for cubic interpolation
            # Create the interpolators for each line
            interpolators = [
                interp.CubicSpline(
                    self.x_points[-look_back:],
                    values[-look_back:],
                    extrapolate=True
                ) for values in self.list_of_values
            ]

            approximation_array = np.zeros((len(interpolators), _point.size))
            for i, interpolator in enumerate(interpolators):
                approximation_array[i, :] = interpolator(x_val)

            dists = approximation_array - _point[None, :]
            np.power(dists, 2, out=dists)
            np.sqrt(dists, out=dists)

            added = []      # for debug purposes
            for i, p in enumerate(_point):
                j = np.min(np.argmin(dists, axis=1))
                dists[:, j] = 1e9
                self.list_of_values[j].append(p)
                if j in added:
                    from IPython import embed; embed()
                    assert False
                added.append(j)

            for i in filter(lambda x: x not in added, range(len(self.list_of_values))):
                self.list_of_values[i].append(0)        # This is a sketchy guardian value

            # Compare with np.asarray(point)

        else:
            for i in range(self.num_lines):
                self.list_of_values[i].append(point[i])
        self.x_points.append(x_val)


def test1():
    t = np.linspace(0, 2*np.pi, 100)

    cos_array = np.cos(4*t)
    sin_array = np.sin(2*t)

    accumulator = PointAccumulator(2)

    for i, (x, (c, s)) in enumerate(zip(t, zip(cos_array, sin_array))):
        accumulator.add_point(x, (c, s), look_back=4)

    ca, sa = accumulator.list_of_values
    x = accumulator.x_points

    fig, ax = plt.subplots(1)
    ax.plot(t, cos_array, color="r", label="cos reference")
    ax.plot(t, sin_array, color="b", label="sin reference")

    ax.plot(x, ca, "--", color="r", label="cos")
    ax.plot(x, sa, "--", color="b", label="sin")

    ax.legend()
    plt.show()


def test2(N=10):
    t = np.linspace(0, N, N + 1)
    series1 = np.random.random(N + 1)
    series2 = np.random.random(N + 1)

    acc = PointAccumulator(num_lines=2)

    for x, (s1, s2) in zip(t, zip(series1, series2)):
        acc.add_point(x, (s1, s2))


def test3():
    t = np.linspace(0, 2*np.pi, 1000)

    series1 = np.cos(t)
    series2 = np.sin(t)
    series3 = np.sin(np.cos(t)*t)
    series3 = np.sin(np.sin(t)*t)
    series4 = np.sin(t)*np.cos(np.sqrt(t))

    acc = PointAccumulator(num_lines=4)
    for x, p in zip(t, zip(series1, series2, series3, series4)):
        acc.add_point(x, p, look_back=20)


    fig, ax = plt.subplots(1)
    ax.plot(t, series1, color="b", label="s1")
    ax.plot(t, series2, color="g", label="s2")
    ax.plot(t, series3, color="r", label="s3")
    ax.plot(t, series4, color="m", label="s4")

    ap1, ap2, ap3, ap4 = acc.list_of_values

    ax.plot(t, ap1, "--", color="b", label="a1")
    ax.plot(t, ap2, "--", color="g", label="a2")
    ax.plot(t, ap3, "--", color="r", label="a3")
    ax.plot(t, ap4, "--", color="m", label="a4")


    ax.legend()
    plt.show()


if __name__ == "__main__":
    # test2(N=1000)
    test3()
