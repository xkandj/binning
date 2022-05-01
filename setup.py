from pathlib import Path

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

version = Path("binning/version.py").read_text(encoding="utf-8")
version_dict = {}
exec(version, version_dict)

setuptools.setup(
    name="binning",
    version=version_dict["__version__"],
    author="xkandj",
    author_email="lliu606@hotmail.com",
    description="feature bin",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/xkandj/binning",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    keywords="feature bin",
    python_requires='>=3.7',
    install_requires=[
        "pandas>=1.3",
        "numpy>=1.18",
        "joblib>=0.13",
        "concurrent_log>=1.0"
    ]
)
