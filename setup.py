from setuptools import (
    setup,
    find_packages,
)


setup(
    name="digeeg",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=["opencv-python", "numpy", "shapely", "matplotlib", "imutils"]
)
