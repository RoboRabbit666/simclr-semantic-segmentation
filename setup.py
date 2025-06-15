from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="simclr-semantic-segmentation",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Self-supervised semantic segmentation using SimCLR pretraining",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/simclr-semantic-segmentation",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest>=6.0", "black>=21.0", "flake8>=3.9", "isort>=5.9"],
        "docs": ["sphinx>=4.0", "sphinx-rtd-theme>=0.5"],
    },
)