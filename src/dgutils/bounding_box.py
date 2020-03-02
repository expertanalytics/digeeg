import numpy as np

import typing as tp

from pathlib import Path

import logging
import os


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


def compute_bounding_box(input_array: np.ndarray) -> tp.Tuple[float, float, float, float]:
    """Compute the bounding box of an array of 2d points."""
    assert input_array.shape[1] == 2, f"Expecting 2D arrays, got {input_array.shape}"
    xmin = np.min(input_array[:, 0])
    xmax = np.max(input_array[:, 0])
    ymin = np.min(input_array[:, 1])
    ymax = np.max(input_array[:, 1])
    return xmin, xmax, ymin, ymax


def filter_intersects_box(
        box: tp.Tuple[float, float, float, float],
        array_of_boxes: np.ndarray,
        indices: np.ndarray
) -> np.ndarray:
    """Return boxes that do NOT intersect with `box`."""
    left_of = box[1] < array_of_boxes[indices, 0]
    right_of = box[0] > array_of_boxes[indices, 1]
    horisontal = right_of | left_of

    above_of = box[2] > array_of_boxes[indices, 3]
    below_of = box[3] < array_of_boxes[indices, 2]
    vertical = above_of | below_of
    return indices[horisontal | vertical]


def concatenate_arrays(array_list: tp.Iterable[np.ndarray]) -> np.ndarray:
    return np.concatenate(array_list)


def sort_bounding_boxes(list_of_time_series: tp.Sequence[np.ndarray]) -> tp.List[np.ndarray]:
    """Group a list of time series into two consecutive non-overlapping series.

    A time series is an array of point (x, y), that is [(x0, y0), (x1, y1), ...].

    The time series are joined based on consecutive non-overlapping bounding boxes.

    Return tuple of array of indices referring to the original sequence of time series.
    """
    # compute_bounding_boxes and convert to arrays
    bounding_boxes = np.asarray(
        [np.array(compute_bounding_box(arr)) for arr in list_of_time_series]
    )

    # Sort the bounding boxes by xmin -- boxes are (xmin, xmax, ymin, ymax)
    xmin_indices = np.argsort(bounding_boxes[:, 0])
    sorted_boxes = bounding_boxes[xmin_indices]

    boxes = []      # Store indices to the joined time series
    current_box_index = 0       # first time series, they are sorted.
    boxes.append(current_box_index)

    # Used fore reference back to the boxes when working with boolean indices
    box_indices = np.arange(xmin_indices.size)

    while True:     # find lowest x-value greater than the xmax of first box
        # all elements greater than -- boolean indices
        greater_than = sorted_boxes[:, 0] > sorted_boxes[current_box_index][1]

        # Integer indices of boxes to the right of the current one
        greater_than_indices = box_indices[greater_than]        # real indices

        not_intersecting_indices = filter_intersects_box(       # real indices
            sorted_boxes[current_box_index],
            sorted_boxes,
            greater_than_indices
        )
        if not_intersecting_indices.size == 0:
            break

        # first element greater than -- real index
        first_greater = np.searchsorted(greater_than[not_intersecting_indices], True)
        current_box_index = not_intersecting_indices[first_greater]     # real index
        boxes.append(current_box_index)

    boxes = np.asarray(boxes)       # TODO: Check that it is sorted
    sequence1 = boxes
    all_indices = np.arange(len(list_of_time_series))
    sequence2 = np.setdiff1d(all_indices, boxes)

    return xmin_indices[sequence1], xmin_indices[sequence2]


def read_number_of_traces(directory: Path) -> np.ndarray:
    matching_files = list(directory.glob("trace*.npy"))
    trace_indices = np.arange(len(matching_files))
    return trace_indices


if __name__ == "__main__":
    t1 = np.array([[1, 2], [2, 4], [3, 1]])
    t2 = np.array([[2, 30], [3, 60], [4, 10]])
    t3 = np.array([[4, 3], [6, 6], [8, -1]])
    t4 = np.array([[7, 13], [6, 11], [8, 7]])

    sort_bounding_boxes([t1, t2, t3, t4])
