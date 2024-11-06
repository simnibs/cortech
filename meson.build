project(
    'Cortech',
    'cpp', 'cython',
    version: '0.1',
    meson_version: '>= 1.2',
    default_options: [
        'buildtype=debugoptimized',
        'b_ndebug=if-release',
        # 'c_std=c99',
        # 'cpp_std=c++14',
    ],
)

valid_host_os = ['darwin', 'linux', 'windows']

# build equivalents
host_os = host_machine.system()             # e.g., linux, darwin, windows
host_cpu_family = host_machine.cpu_family() # e.g., x64_86
host_cpu = host_machine.cpu()               # e.g., amd64

assert(valid_host_os.contains(host_os), 'Unsupported OS')

# ==== COMPILERS ====

cpp = meson.get_compiler('cpp')
cython = meson.get_compiler('cython')

# ==== PROGRAMS ====

# (generator() does not accept compilers, only programs)
cython_prog = find_program(cython.cmd_array()[0])

# GLOBAL compiler arguments

# if host_os == 'linux'
#
# elif host_os == 'darwin'
#
# elif host_os == 'windows'
#
# endif

# GCC options
# warnings      https://gcc.gnu.org/onlinedocs/gcc/Warning-Options.html
# optimization  https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html

# ==== COMPILER ARGUMENTS ====

# When buildtype=debug, meson automatically sets things like -g and -Wall

# NumPy and Boost cause various warnings. Suppress them
cython_cpp_args = cpp.get_supported_arguments(
    '-Wno-cpp',
    '-Wno-nonnull',
    '-Wno-sign-compare',
    '-Wno-unused-value',
)
add_project_arguments(cython_cpp_args, language: 'cython')


# ==== GENERATORS ====

# meson ignores the flags in the .pyx files so specify here
cython_cplus_args = [
    '-3',
    '--cplus',
    '--fast-fail',
    '--output-file', '@OUTPUT@',
    '@INPUT@',
]
cython_gen_cpp = generator(
    cython_prog,
    arguments: cython_cplus_args,
    output: '@BASENAME@.cpp',
)

# ==== DEPENDENCIES ====

# Get python installation
python_module = import('python')
python = python_module.find_installation(pure: false)
python_dep = python.dependency()
# python.get_install_dir() is the dir which `subdir` is relative to, i.e.,
# .../site-packages/. Therefore
cortech_install_dir = python.get_install_dir()

# NumPy dependency
incdir_numpy = run_command(
    python,
    ['-c', 'import numpy; print(numpy.get_include())'],
    check: true,
).stdout().strip()
numpy_include = include_directories(incdir_numpy)

numpy_dep = declare_dependency(include_directories: numpy_include)
cgal_dep = dependency('cgal', version: '>= 5.5')

# files = ['3RD-PARTY.md', 'LICENSE.txt', 'README.md']
# install_data(files, install_dir: simnibs_install_dir / 'simnibs')

subdir('cortech')
