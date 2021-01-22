from setuptools import setup

from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir

import sys
import os

__version__ = "0.0.1"


ext_modules = [
    Pybind11Extension("poselib",["pyposelib.cpp"],
    extra_objects=[os.path.join(os.getenv('CONDA_PREFIX'),'lib','libPoseLib.a')]),
]

setup(
    name="poselib",
    include_dirs=[os.path.join(os.getenv('CONDA_PREFIX'),'include'),
                  os.path.join(os.getenv('CONDA_PREFIX'),'include','eigen3')],
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)