import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="utils_python",
    version="1.0.0",
    author="wook3910",
    author_email="wook3910@gmail.com",
    description="a small utilities for python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wk39/utils_python",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
