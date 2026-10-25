# setup.py: builds the C++ module for D11 (done).
#   Windows: install "Desktop development with C++" (Visual Studio Build Tools) first.
#   python setup.py build_ext --inplace
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext = Pybind11Extension("fastmatch", ["cpp/fastmatch.cpp"], cxx_std=17)
setup(name="fastmatch", version="0.1", ext_modules=[ext], cmdclass={"build_ext": build_ext})
