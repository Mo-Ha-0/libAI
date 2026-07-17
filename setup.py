from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11
import sys

extra_compile_args = ["-O2", "-march=native", "-ffast-math", "-fopenmp"]
extra_link_args = ["-fopenmp"]

import platform
if platform.system() == "Linux":
    extra_link_args.append("-lstdc++")

ext_modules = [
    Pybind11Extension(
        "libAI._libAI_core",
        [
            "bindings/bindings.cpp",
            "src/layers/dense.cpp",
            "src/layers/activations.cpp",
            "src/layers/losses.cpp",
            "src/layers/regularization.cpp",
            "src/optimizers/optimizers.cpp",
            "src/network.cpp",
            "src/trainer.cpp",
            "src/tuning.cpp",
        ],
        include_dirs=["include", pybind11.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++17",
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
