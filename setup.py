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
        "scipy"
    ],
    entry_points={
        "console_scripts": [
            "split-image = scripts.split_image:main",
            "segment-traces = scripts.segment_traces:main",
            "segment-traces-bw = scripts.segment_traces_bw:main",
            "digitise-traces = scripts.digitise_traces:main",
            "plot-trace = scripts.plot_array:main",
        ]
    }
)
