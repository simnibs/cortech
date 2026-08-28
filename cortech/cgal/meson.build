cgal_dir = 'cortech' / 'cgal'

python_sources = [
    '__init__.py',
]
python.install_sources(python_sources, subdir: cgal_dir)

cpp_cgal_macros = [
    '-DCGAL_EIGEN3_ENABLED'#, '-DCGAL_PMP_USE_CERES_SOLVER'
]
# PMP area smoothing needs this
# -DCGAL_PMP_USE_CERES_SOLVER

# the current source (and build) dir is included by default; for header files
# and source files
cgal_helpers_dep = declare_dependency(
    # include_directories: include_directories('.'),
sources: 'cgal_helpers.cpp',
)

aabb_tree = python.extension_module(
    'aabb_tree',
    cython_gen_cpp.process('aabb_tree.pyx'),
    cpp_args: [cpp_cgal_macros],
    dependencies: [numpy_dep, cgal_dep, cgal_helpers_dep],
    install: true,
    subdir: cgal_dir,
)

alpha_wrap_3 = python.extension_module(
    'alpha_wrap_3',
    cython_gen_cpp.process('alpha_wrap_3.pyx'),
    cpp_args: [cpp_cgal_macros],
    dependencies: [numpy_dep, cgal_dep, cgal_helpers_dep],
    install: true,
    subdir: cgal_dir,
)

# constrained_triangulations = python.extension_module(
#     'constrained_triangulations',
#     cython_gen_cpp.process('constrained_triangulations.pyx'),
#     cpp_args: [cpp_cgal_macros],
#     dependencies: [numpy_dep, cgal_dep, cgal_helpers_dep],
#     install: true,
#     subdir: cgal_dir,
# )

pmp = python.extension_module(
    'boost_graph',
    cython_gen_cpp.process('boost_graph.pyx'),
    cpp_args: [cpp_cgal_macros],
    dependencies: [numpy_dep, cgal_dep, cgal_helpers_dep], # ceres_dep
    install: true,
    subdir: cgal_dir,
)

convex_hull_3 = python.extension_module(
    'convex_hull_3',
    cython_gen_cpp.process('convex_hull_3.pyx'),
    cpp_args: [cpp_cgal_macros],
    dependencies: [numpy_dep, cgal_dep, cgal_helpers_dep],
    install: true,
    subdir: cgal_dir,
)

mesh_3 = python.extension_module(
    'mesh_3',
    cython_gen_cpp.process('mesh_3.pyx'),
    cpp_args: [cpp_cgal_macros],
    dependencies: [numpy_dep, cgal_dep, cgal_helpers_dep],
    install: true,
    subdir: cgal_dir,
)

pmp = python.extension_module(
    'polygon_mesh_processing',
    cython_gen_cpp.process('polygon_mesh_processing.pyx'),
    cpp_args: [cpp_cgal_macros],
    dependencies: [numpy_dep, cgal_dep, cgal_helpers_dep], # ceres_dep
    install: true,
    subdir: cgal_dir,
)

sms = python.extension_module(
    'surface_mesh_simplification',
    cython_gen_cpp.process('surface_mesh_simplification.pyx'),
    cpp_args: [cpp_cgal_macros],
    dependencies: [numpy_dep, cgal_dep, cgal_helpers_dep],
    install: true,
    subdir: cgal_dir,
)

sms = python.extension_module(
    'surface_mesh_skeletonization',
    cython_gen_cpp.process('surface_mesh_skeletonization.pyx'),
    cpp_args: [cpp_cgal_macros],
    dependencies: [numpy_dep, cgal_dep, cgal_helpers_dep],
    install: true,
    subdir: cgal_dir,
)

tetrahedral_remeshing = python.extension_module(
    'tetrahedral_remeshing',
    cython_gen_cpp.process('tetrahedral_remeshing.pyx'),
    cpp_args: [cpp_cgal_macros],
    dependencies: [numpy_dep, cgal_dep, cgal_helpers_dep],
    install: true,
    subdir: cgal_dir,
)
