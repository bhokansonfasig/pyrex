from setuptools import setup
import os.path

# Grab information about package without loading it
about = {}
with open(os.path.join('pyrex', '__about__.py')) as f:
    exec(f.read(), about)

setup(
    name = about["__fullname__"].lower(),
    version = about["__version__"],
    description = about["__description__"],
    long_description = about["__long_description__"],
    classifiers = [
        "Development Status :: 5 - Production/Stable",
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
    # Not using find_packages since it clashes with PEP 420 use
    packages = ['pyrex', 'pyrex.custom', 'pyrex.custom.irex',
                'pyrex.custom.ara', 'pyrex.custom.arianna',
                'pyrex.custom.layered_ice',],
    python_requires = '>= 3.6',
    install_requires = [
        'numpy>=1.17',
        'scipy>=1.4',
        'h5py>=3.0',
    ],
    setup_requires = ['pytest-runner'],
    tests_require = ['pytest'],
    package_data = {
        '': ['README.rst', 'LICENSE', 'PyREx Documentation.pdf'],
        'pyrex': ['data/secondary/*/*.vec'],
        'pyrex.custom.ara': ['data/*.txt', 'data/*.sqlite'],
        'pyrex.custom.arianna': ['data/*.txt', 'data/*.csv', 'data/*.tar.gz']
    },
)
