import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def conda_pkgconfig_generator(out_dir=None):
    """Generate minimal pkg-config files for conda packages boost (boost-cpp),
    eigen, and cgal (cgal-cpp).

    I have hardcoded dependencies etc. which is not ideal but should work.

    pkg-config files are automatically generated for some packages, e.g.,
    gmp, mpfr, mpich, tbb, zlib

    Suppose we install .pc files to deps then use

        pip install -e . -C pkg-config-path=deps

    when building.

    EXAMPLE

    prefix=/usr/local
    exec_prefix=${prefix}
    includedir=${prefix}/include
    libdir=${exec_prefix}/lib

    Name: foo
    Description: The foo library
    Version: 1.0.0
    Requires: bar
    Cflags: -I${includedir}/foo
    Libs: -L${libdir} -lfoo

    """
    assert "CONDA_PREFIX" in os.environ, "This is a convenience function for generating pkg-config files for libraries in a CONDA installation."
    prefix = Path(os.environ["CONDA_PREFIX"])

    # dir where conda stores pkg-config files (.pc) from packages here
    pkgconfig_dir = prefix / "lib" / "pkgconfig"

    prefix = prefix
    if sys.platform == "win32":
        exec_prefix = prefix / "Lib"
    else:
        exec_prefix = prefix

    out_dir = pkgconfig_dir if out_dir is None else Path(out_dir)
    if not out_dir.exists():
        out_dir.mkdir()

    print(f"Saving to {out_dir}")
    # ignore = ["libgcc-ng", "libstdcxx-ng"]
    # strip = ["cpp"]

    prefixes = dict(prefix=prefix, exec_prefix=exec_prefix)
    packages = dict(
        boost = prefixes | dict(inc_subdir="include"),
        cgal = prefixes | dict(inc_subdir="include", requires = "boost >= 1.74 eigen >= 3.4 mpfr >= 4.2"),
        eigen = prefixes | dict(inc_subdir = "include/eigen3"),
        # tbb = prefixes | dict(lib_subdir="lib", inc_subdir="include", libs=["tbb", "tbbmalloc"])
    )

    pkg_config = {}
    for name, settings in packages.items():
        # if pkg_exists(pkgconfig_dir, name):
        #     print(f"pkg-config already exists for package {name}. Skipping.")
        #     continue
        settings["version"] = get_package_version(name)
        pkg_config[name] = make_pkg_config(name, **settings)

    for name, content in pkg_config.items():
        # print(content)
        with open(out_dir / f"{name}.pc", "w") as f:
            f.write(content)

    if len(pkg_config) > 0:
        print(f"Wrote pkg-config files for\n")
        for name, settings in packages.items():
            print(f"    {name:10s} {settings['version']:10s}")
        print()


def make_pkg_config(
    name: str,
    version: str,
    prefix: Path | str,
    exec_prefix: None | Path | str = None,
    description: str = "",
    requires: str = "",
    lib_subdir: None | str = None,
    inc_subdir: None | str = None,
    bin_subdir: None | str = None,
    libs: None | list[str] | tuple[str] = None,
    cflags: str = "",
):
    description = description or f"conda package: {name}"
    exec_prefix = prefix if exec_prefix is None else exec_prefix

    out = "\n".join([f"prefix={prefix}", f"exec_prefix={exec_prefix}"])

    if lib_subdir:
        out += f"\nlibdir=${{exec_prefix}}/{lib_subdir}"
    if inc_subdir:
        out += f"\nincludedir=${{prefix}}/{inc_subdir}"
    if bin_subdir:
        out += f"\nbindir=${{exec_prefix}}/{bin_subdir}"
    out += "\n".join(["\n", f"Name: {name:s}", f"Description: {description:s}", f"Version: {version:s}"])
    if requires:
        out += f"\nRequires: {requires:s}"
    if lib_subdir or libs:
        out += f"\nLibs:"
        if lib_subdir:
            out += f" -L${{libdir}}"
        if libs:
            out += " " + " ".join([f"-l{lib}" for lib in libs])
    if inc_subdir or cflags:
        out +=  f"\nCflags:"
        if inc_subdir:
            out += f" -I${{includedir}}"
        if cflags:
            out += f" {cflags}"
    out += "\n"
    return out


def pkg_exists(pkgconfig_dir, name):
    return (pkgconfig_dir / name).with_suffix(".pc").exists()


def get_package_version(name):
    out = subprocess.run(f"conda list {name} --json".split(), capture_output=True)
    info = json.loads(out.stdout.decode())
    if len(info) == 0:
        raise RuntimeError(
            f"Could not find conda package: {name}. conda list returned\n{out.stdout.decode()}"
        )
    elif len(info) > 1:
        if name == "tbb":
            info = info[:1] # tbb and tbb-devel so choose tbb
        elif name == "boost":
            info = info[:1] # libboost, libboost-devel, libboost-headers
        else:
            raise RuntimeError(
                f"Found multiple conda packages for {name}. conda list returned\n{out.stdout.decode()}"
            )
    assert len(info) == 1
    info = info[0]
    return info["version"]


def parse_args(argv):
    parser = argparse.ArgumentParser(
        "generate_pkg",
        description="Generate .pc files for boost, eigen, and CGAL when installed from conda."
    )
    parser.add_argument(
        "-d", "--out-dir", help="Directory in which to store the generate .pc files."
    )
    return parser.parse_args(argv)

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    conda_pkgconfig_generator(args.out_dir)