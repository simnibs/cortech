# cortech
A suite of tools for cortical layer placement and analysis.

## Installation

This package requires compiling some CGAL functions. You can install CGAL using any method you prefer; meson (which we use for building the package) just needs to locate CGAL with pkg-config.. Here we describe the installation using Conan or Conda for managing the c++ dependencies.

### Using Conan

First

    pip install conan
    conan profile detect
    conan install --build=missing --output-folder=$PWD/conan_deps .

Then

    pip install --no-build-isolation --no-deps --config-settings=setup-args=--native-file=$PWD/conan_deps/conan_meson_native.ini -e .

### Using Conda
If you are using `conda` to manage build dependencies, then start by running the following script to generate the necessary pkg config files

    python tools/generate_pkgs.py

This will generate pkg files for Boost, CGAL, Eigen, and TBB (TBB already exists but we want to include tbbmalloc in addition to tbb) and save them in `$CONDA_PREFIX/lib/pkgconfig` which is where conda stores its pkg config files (an alternative output directory can also be specified).

**Note** This path probably needs to be adjusted on windows!

Next, for an editable (developer) installation, use

    pip install --no-build-isolation --no-deps -e .
