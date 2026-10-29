python_sources = [
    '__init__.py',
    'complex.py',
    'constants.py',
    'cortex.py',
    'graph.py',
    'polyline.py',
    'sphere_utils.py',
    'surface.py',
    'utils.py',
    'visualization.py',
    'interactive_visualization.py',
]
python.install_sources(python_sources, subdir: 'cortech')

aabb_tree = python.extension_module(
    'utils_cpp',
    cython_gen_cpp.process('utils_cpp.pyx'),
    dependencies: [numpy_dep],
    install: true,
    subdir: 'cortech',
)

subdir('cgal')
subdir('freesurfer')
subdir('models')
subdir('bin')
