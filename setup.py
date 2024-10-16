import os
from setuptools import setup, find_packages
from setuptools.command.install import install

setup(
    package_dir={"": "src"},  # This points setuptools to the src directory
    packages=["poliprompt"],  # This is your main package in src/poliprompt
)
python_requires=">=3.10"
