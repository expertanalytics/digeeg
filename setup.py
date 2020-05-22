from setuptools import (
    setup,
    find_packages,
)


setup(
    name="digeeg",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "opencv-python",
        "numpy",
        "shapely",
        "matplotlib",
        "imutils",
        "pylint",
        "scipy",
        "h5py"
    ],
    entry_points={
        "console_scripts": [
            "split-image = scripts.split_image:main",
            "segment-traces = scripts.segment_traces:main",
            "segment-traces-bw = scripts.segment_traces_bw:main",
            "digitise-traces = scripts.digitise_traces:main",
            "plot-trace = scripts.plot_array:main",
            "visualise = scripts.visualise:main",
            "segment-digitise = scripts.segment_digitise:main",
            "join-traces = scripts.join_traces:main",
            "check-traces = scripts.join2:main",
        ]
    }
)
