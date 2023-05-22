import os

from setuptools import find_packages, setup


def get_version():
    init_py_path = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "zsvcip", "__init__.py"
    )
    init_py = open(init_py_path, "r").readlines()
    version_line = [l.strip() for l in init_py if l.startswith("__version__")][0]
    version = version_line.split("=")[-1].strip().strip("'\"")
    return version


setup(
    name="zsvcip",
    version=get_version(),
    author="Seiya Ito <ito.seiya@vss.it.aoyama.ac.jp>",
    url="https://github.com/seiyaito/vcip",
    description="Unofficial implementation of Zero-shot Visual Commonsense Immorality Prediction.",
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=[
        "easydict",
        "polars",
        "pyyaml",
        "torchmetrics",
        "transformers",
    ],
)
