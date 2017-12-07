from setuptools import setup, find_packages
import os.path

# Grab information about package without loading it
about = {}
with open(os.path.join('pyrex', '__about__.py')) as f:
    exec(f.read(), about)

setup(
    name = about["__fullname__"],
    version = about["__version__"],
    description = about["__description__"],
    long_description = about["__long_description__"],
    classifiers = [
        "Development Status :: 4 - Beta",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    keywords = "radio neutrino astronomy physics",
    url = about["__url__"],
    author = about["__author__"],
    author_email = about["__author_email__"],
    license = about["__license__"],
    packages = find_packages(),
    python_requires = '>= 3.6',
    install_requires = [
        'numpy>=1.13',
        'scipy>=0.19'
    ],
    setup_requires = ['pytest-runner'],
    tests_require = ['pytest'],
    package_data = {
        '': ['README.rst', 'LICENSE', 'PyREx Documentation.pdf',
             'Code Examples.ipynb', 'PyREx Demo.ipynb'],
        'pyrex': [],
    },
)