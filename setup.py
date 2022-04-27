import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="binning",
    version="beta",
    author="xiazhou",
    author_email="lliu606@hotmail.com",
    description="feature bin",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/xkandj/binning",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
