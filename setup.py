#!/usr/bin/env python3
"""
Setup script for the Optimization Framework package.

This package provides a comprehensive OOP optimization framework designed for
ARM64 Windows environments where scipy is unavailable. It includes multiple
variable types, constraint handling, and various optimization algorithms.
"""

import os
import sys
from pathlib import Path
from setuptools import setup, find_packages

# Ensure we're using Python 3.7+
if sys.version_info < (3, 7):
    raise RuntimeError("This package requires Python 3.7 or later")

# Get the long description from README file
here = Path(__file__).parent.resolve()
long_description = ""
readme_path = here / "README.md"
if readme_path.exists():
    with open(readme_path, encoding="utf-8") as f:
        long_description = f.read()
else:
    # Fallback description if README doesn't exist yet
    long_description = """
    A comprehensive Object-Oriented Programming (OOP) optimization framework designed 
    specifically for environments where scipy is unavailable (such as ARM64 Windows). 

    Features:
    - Multiple variable types: continuous, integer, binary with bounds
    - Flexible constraint handling with multiple constraint types
    - Multiple optimization algorithms: greedy search, genetic algorithm, simulated annealing
    - NASA-style production asserts for robust validation
    - Complete solution tracking with optimization history
    - Clean, extensible OOP architecture
    """

# Read version from package
def get_version():
    """Extract version from package __init__.py"""
    version_file = here / "optimization_framework" / "__init__.py"
    if version_file.exists():
        with open(version_file, encoding="utf-8") as f:
            for line in f:
                if line.startswith("__version__"):
                    return line.split("=")[1].strip().strip('"').strip("'")
    return "0.1.0"  # Default version

# Define package requirements
install_requires = [
    # Core dependencies - keeping minimal for ARM64 compatibility
    "typing-extensions>=3.7.4; python_version<'3.8'",  # For older Python versions
]

# Optional dependencies for development and testing
extras_require = {
    "dev": [
        "pytest>=6.0.0",
        "pytest-cov>=2.10.0",
        "black>=21.0.0",
        "flake8>=3.8.0",
        "mypy>=0.800",
        "pre-commit>=2.0.0",
    ],
    "test": [
        "pytest>=6.0.0",
        "pytest-cov>=2.10.0",
        "pytest-mock>=3.0.0",
        "hypothesis>=6.0.0",  # For property-based testing
    ],
    "docs": [
        "sphinx>=3.0.0",
        "sphinx-rtd-theme>=0.5.0",
        "sphinx-autodoc-typehints>=1.10.0",
    ],
    "benchmark": [
        "matplotlib>=3.0.0",  # For plotting optimization results
        "pandas>=1.0.0",      # For data analysis
        "memory-profiler>=0.58.0",  # For memory usage analysis
    ]
}

# Add 'all' extra that includes everything
extras_require["all"] = list(set(
    dep for extra_deps in extras_require.values() 
    for dep in extra_deps
))

# Entry points for command-line tools (if any)
entry_points = {
    "console_scripts": [
        # Example: "optimize=optimization_framework.cli:main",
    ],
}

# Package data to include
package_data = {
    "optimization_framework": [
        "py.typed",  # PEP 561 marker for type information
        "*.pyi",     # Type stub files
    ],
}

# Files to exclude from distribution
exclude_package_data = {
    "": [
        "tests/*",
        "test_*",
        "*_test.py",
        "*.pyc",
        "__pycache__/*",
        ".pytest_cache/*",
        ".coverage",
        "*.log",
    ]
}

setup(
    # Basic package information
    name="optimization-framework",
    version=get_version(),
    description="A comprehensive OOP optimization framework for ARM64 Windows environments",
    long_description=long_description,
    long_description_content_type="text/markdown",

    # Author and contact information
    author="Optimization Framework Team",
    author_email="team@optimization-framework.dev",
    maintainer="Optimization Framework Team",
    maintainer_email="team@optimization-framework.dev",

    # URLs and links
    url="https://github.com/optimization-framework/optimization-framework",
    project_urls={
        "Documentation": "https://optimization-framework.readthedocs.io/",
        "Source": "https://github.com/optimization-framework/optimization-framework",
        "Tracker": "https://github.com/optimization-framework/optimization-framework/issues",
        "Changelog": "https://github.com/optimization-framework/optimization-framework/blob/main/CHANGELOG.md",
    },

    # License information
    license="MIT",
    license_files=["LICENSE"],

    # Package discovery and structure
    packages=find_packages(
        exclude=[
            "tests",
            "tests.*",
            "docs",
            "docs.*",
            "examples",
            "examples.*",
            "benchmarks",
            "benchmarks.*",
        ]
    ),
    package_data=package_data,
    exclude_package_data=exclude_package_data,
    include_package_data=True,

    # Python version requirements
    python_requires=">=3.7",

    # Dependencies
    install_requires=install_requires,
    extras_require=extras_require,

    # Entry points
    entry_points=entry_points,

    # Package classification
    classifiers=[
        # Development Status
        "Development Status :: 4 - Beta",

        # Intended Audience
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",

        # Topic Classification
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Libraries :: Application Frameworks",

        # License
        "License :: OSI Approved :: MIT License",

        # Programming Language
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3 :: Only",

        # Operating System
        "Operating System :: OS Independent",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",

        # Environment
        "Environment :: Console",
        "Environment :: Other Environment",

        # Natural Language
        "Natural Language :: English",

        # Typing
        "Typing :: Typed",
    ],

    # Keywords for PyPI search
    keywords=[
        "optimization",
        "mathematical-optimization",
        "genetic-algorithm",
        "simulated-annealing",
        "greedy-search",
        "constraint-optimization",
        "oop",
        "framework",
        "arm64",
        "windows",
        "scipy-alternative",
        "optimization-algorithms",
        "mathematical-programming",
    ],

    # Additional metadata
    platforms=["any"],
    zip_safe=False,  # Don't create zip files for better debugging

    # Options for different build tools
    options={
        "bdist_wheel": {
            "universal": False,  # Not universal since we have platform-specific optimizations
        },
        "egg_info": {
            "tag_build": "",
            "tag_date": False,
        },
    },

    # Test suite configuration
    test_suite="tests",
    tests_require=extras_require["test"],

    # Additional package metadata for modern setuptools
    project_urls={
        "Documentation": "https://optimization-framework.readthedocs.io/",
        "Source Code": "https://github.com/optimization-framework/optimization-framework",
        "Issue Tracker": "https://github.com/optimization-framework/optimization-framework/issues",
        "Changelog": "https://github.com/optimization-framework/optimization-framework/blob/main/CHANGELOG.md",
        "Funding": "https://github.com/sponsors/optimization-framework",
        "Say Thanks!": "https://saythanks.io/to/optimization-framework",
    },
)

# Post-installation message
print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     Optimization Framework Installation                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Thank you for installing the Optimization Framework!                       ║
║                                                                              ║
║  This package provides a comprehensive OOP optimization framework           ║
║  designed for ARM64 Windows environments where scipy is unavailable.       ║
║                                                                              ║
║  Key Features:                                                               ║
║  • Multiple variable types (continuous, integer, binary)                    ║
║  • Flexible constraint handling                                             ║
║  • Multiple optimization algorithms                                         ║
║  • NASA-style production validation                                         ║
║  • Complete solution tracking                                               ║
║                                                                              ║
║  Quick Start:                                                                ║
║    from optimization_framework import *                                      ║
║                                                                              ║
║  Documentation: https://optimization-framework.readthedocs.io/              ║
║  Issues: https://github.com/optimization-framework/issues                   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")