## Scripts

### Split Image

Split the images based on the black square markers such that each part has only two markers. There will be
minor overlap between the parts.

### Extract Traces

Segment all the traces in an EEG image.

### Digitise Traces

Convert a segmented EEG trace to a time series stored as a numpy array.


### Plot Array

A simple utility for plotting an array. The array has to be of the shape (N, 2). This script is intended to be
used with `digitise_traces.py`.
