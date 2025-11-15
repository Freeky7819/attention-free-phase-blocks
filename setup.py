"""
Setup script for Resonant Models package.

Installation:
    pip install -e .  # Development mode
    pip install .     # Standard install
"""

from setuptools import setup, find_packages
import os

# Read long description from README
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="attention-free-phase-blocks",
    version="0.1.0",
    author="Damjan Žakelj",
    description="Attention-free transformer variants with log-periodic phase blocks and compact memory: code and experiments for language modeling and synthetic needle-in-a-haystack retrieval",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Freeky7819/attention-free-phase-blocks",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
        ],
        "wandb": ["wandb>=0.15.0"],
        "tensorboard": ["tensorboard>=2.13.0"],
        "api": [
            "fastapi>=0.104.0",
            "uvicorn[standard]>=0.24.0",
            "pydantic>=2.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "resonant-train=train:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
