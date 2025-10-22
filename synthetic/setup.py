import os
from setuptools import find_packages
from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README.md file
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
  long_description = f.read()

setup(
    name="CHiLD", # Replace with your own username
    version="0.0.1",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires = [
        "wandb>=0.16.0",
        "lightning>=2.0.0",
        "torch>=1.11.0,<4.0",
        "disentanglement-lib>=1.4",
        "torchvision",
        "torchaudio",
        "h5py",
        "ipdb",
        "opencv-python",
        "pymunk",
    ],
    tests_require=[
        "pytest"
    ],
)
