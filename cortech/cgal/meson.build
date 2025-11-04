cgal_dir = 'cortech' / 'cgal'

python_sources = [
    '__init__.py',
]
python.install_sources(python_sources, subdir: cgal_dir)

cpp_cgal_args = cpp.get_supported_arguments(
    '-Os',              # optimize for size
    '-flto=auto',       # apply standard link-time optimizer
    '-frounding-math',  # Disable optimizations related to float rounding
)

# if host_os == 'linux'
# elif host_os == 'darwin'
if host_os == 'windows'
    win_args = cpp.get_supported_arguments(
        '/bigobj',      # allow large obj files
    )
    cpp_cgal_args += win_args
endif

cpp_cgal_macros = ['-DCGAL_EIGEN3_ENABLED']
# PMP area smoothing needs this
# -DCGAL_PMP_USE_CERES_SOLVER

# the current source (and build) dir is included by default; for header files
# and source files
cgal_helpers_dep = declare_dependency(
    # include_directories: include_directories('.'),
    sources: 'cgal_helpers.cpp',
)

pmp = python.extension_module(
    'aabb_tree',
    cython_gen_cpp.process('aabb_tree.pyx'),
    cpp_args: [cython_cpp_args, cpp_cgal_args, cpp_cgal_macros],
    dependencies: [numpy_dep, cgal_dep, cgal_helpers_dep],
    install: true,
    subdir: cgal_dir,
)

pmp = python.extension_module(
    'alpha_wrap_3',
    cython_gen_cpp.process('alpha_wrap_3.pyx'),
    cpp_args: [cython_cpp_args, cpp_cgal_args, cpp_cgal_macros],
    dependencies: [numpy_dep, cgal_dep, cgal_helpers_dep],
    install: true,
    subdir: cgal_dir,
)

pmp = python.extension_module(
    'convex_hull_3',
    cython_gen_cpp.process('convex_hull_3.pyx'),
    cpp_args: [cython_cpp_args, cpp_cgal_args, cpp_cgal_macros],
    dependencies: [numpy_dep, cgal_dep, cgal_helpers_dep],
    install: true,
    subdir: cgal_dir,
)

pmp = python.extension_module(
    'polygon_mesh_processing',
    cython_gen_cpp.process('polygon_mesh_processing.pyx'),
    cpp_args: [cython_cpp_args, cpp_cgal_args, cpp_cgal_macros],
    dependencies: [numpy_dep, cgal_dep, cgal_helpers_dep], # ceres_dep
    install: true,
    subdir: cgal_dir,
)

pmp = python.extension_module(
    'surface_mesh_simplification',
    cython_gen_cpp.process('surface_mesh_simplification.pyx'),
    cpp_args: [cython_cpp_args, cpp_cgal_args, cpp_cgal_macros],
    dependencies: [numpy_dep, cgal_dep, cgal_helpers_dep],
    install: true,
    subdir: cgal_dir,
)
