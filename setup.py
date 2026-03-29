"""
FLUXION - Physics-Native Silicon Intelligence
A Thermodynamic Placement Engine for Open-Access Chip Design
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="fluxion",
    version="1.0.0",
    author="FLUXION Project",
    author_email="fluxion@example.com",
    description="Physics-Native Chip Placement Engine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fluxion-project/fluxion",
    packages=find_packages(where="src/python"),
    package_dir={"": "src/python"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Electronic Design Automation (EDA)",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: C++",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
    ],
    extras_require={
        "gpu": ["pyopencl>=2021.1"],
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "isort>=5.0",
            "mypy>=0.900",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "fluxion=fluxion.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)