#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from setuptools import setup, find_packages
#from setuptools import setup
from distutils.core import setup, Extension


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="kypackage",
    version="5.0.1",
    author="ky",
    author_email="ky@gmail.com",
    description="This is a simple example package for demo.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kyang4881/KyangGitHub/tree/master/Package%20Setup",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.6",
    ext_package='kypackage',
    ext_modules=[Extension('first_script', ['first_script.first_function']),
                 Extension('second_script', ['second_script.second_function']),
                 Extension('math.add_script', ['add_script.add_function']),
                 Extension('math.multiply_script', ['multiply_script.multiply_function']),
                 Extension('quantity.quantity_script', ['quantity_script.quantity_function']),],
    packages=find_packages(),
 
)
