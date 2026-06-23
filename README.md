# cortech
A suite of tools for cortical layer placement and analysis.

## Installation From Source

This package requires compiling some CGAL functions. You can install CGAL using any method you prefer; meson (which we use for building the package) just needs to locate CGAL with pkg-config. Here we describe the installation using Conan or Conda for managing the c++ dependencies.

### Using Conan

First (using linux as example)

    pip install conan
    conan config install -tf profiles conan_profiles
    conan install --profile:all linux --build=missing --output-folder=$PWD/conan_deps .

Then

    pip install --config-settings=setup-args=--native-file=$PWD/conan_deps/conan_meson_native.ini --no-build-isolation --no-deps -e .

### Using Conda
If you are using `conda` to manage build dependencies, then start by running the following script to generate the necessary pkg config files

    python tools/generate_pkgs.py

This will generate pkg files for Boost, CGAL, Eigen, and TBB (TBB already exists but we want to include tbbmalloc in addition to tbb) and save them in `$CONDA_PREFIX/lib/pkgconfig` which is where conda stores its pkg config files (an alternative output directory can also be specified).

**Note** This path probably needs to be adjusted on windows!

You must also install pkg-config from conda as pkg-config provided by the system (e.g., in /bin/) will not search this particular directory

    conda install pkg-config

Next, for an editable (developer) installation, use

    pip install --no-build-isolation --no-deps -e .
