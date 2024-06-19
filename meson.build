project(
    'qbindiff',
    'cython', 'c', 'cpp',
    version: '1.2.2',
    license: 'Apache-2.0',
    license_files: 'LICENSE',
    meson_version: '>= 1.1.0',
)

py = import('python').find_installation(pure: false)
py_deps = py.dependency()

# NumPy include directory - needed in all submodules
# The try-except is needed because when things are split across drives on
# Windows, there is no relative path and an exception gets raised. There may be
# other such cases, so add a catch-all and switch to an absolute path. Relative
# paths are needed when for example a virtualenv is placed inside the source
# tree; Meson rejects absolute paths to places inside the source tree.
# For cross-compilation it is often not possible to run the Python interpreter
# in order to retrieve numpy's include directory. It can be specified in the
# cross file instead:
#   [properties]
#   numpy-include-dir = /abspath/to/host-pythons/site-packages/numpy/core/include
#
# This uses the path as is, and avoids running the interpreter.
incdir_numpy = meson.get_external_property('numpy-include-dir', 'not-given')
if incdir_numpy == 'not-given'
    incdir_numpy = run_command(
        py,
        [
            '-c',
            '''import os
import numpy as np
try:
    incdir = os.path.relpath(np.get_include())
except Exception:
    incdir = np.get_include()
print(incdir)'''
        ],
        check: true
    ).stdout().strip()
endif

inc_np = include_directories(incdir_numpy)
np_dep = declare_dependency(include_directories: inc_np)

omp = dependency(
    'openmp',
    not_found_message: 'OpenMP not found, QBinDiff will be compiled without support for parallelization',
    required: false
)

cython_gen_cpp = generator(
    find_program('cython'),
    arguments: ['-3', '--cplus', '--fast-fail', '-o', '@OUTPUT@', '-I', '@BUILD_ROOT@', '@INPUT@'],
    output: '@BASENAME@.cpp'
)

cpp_default_args = ['-O3']

subdir('src/qbindiff')
