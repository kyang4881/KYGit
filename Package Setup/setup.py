#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import setuptools
from setuptools import setup


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="kypackage",
    version="3.0.0",
    author="ky",
    author_email="kyang4881@gmail.com",
    description="example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kyang4881/KyangGitHub/tree/master/Package%20Setup",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.6",
)

