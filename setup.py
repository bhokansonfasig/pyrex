from setuptools import setup, find_packages
import pyrex

setup(
    name = "PyREx",
    version = pyrex.__version__,
    description = pyrex.__doc__.splitlines()[0],
    long_description = pyrex.__doc__,
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
    url = "https://github.com/bhokansonfasig/pyrex",
    author = "Ben Hokanson-Fasig",
    author_email = "fasig@icecube.wisc.edu",
    license = "MIT",
    packages = find_packages(),
    python_requires = '>= 3.5',
    install_requires = [
        'numpy',
        'scipy'
    ],
    setup_requires = ['pytest-runner'],
    tests_require = ['pytest'],
    package_data = {
        '': ['README.rst', 'LICENSE', 'PyREx Documentation.pdf',
             'Code Examples.ipynb', 'PyREx Demo.ipynb'],
        'pyrex': [],
    },
)