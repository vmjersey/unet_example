import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="eidetic_ml",
    version="0.0.2",
    author="James L. Barker",
    author_email="vmjersey@hotmail.com",
    description="Several Classes and Functions for Machine Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vmjersey/eidetic_ml",
    packages=['eidetic_ml'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)


