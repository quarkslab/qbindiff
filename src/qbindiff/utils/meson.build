python_sources = [files(
    '__init__.py',
    'utils.py',
)]

py.install_sources(
    python_sources,
    subdir: 'qbindiff/utils'
)

py.extension_module(
    'openmp_helpers',
    cython_gen_cpp.process('openmp_helpers.pyx'),
    override_options: ['cython_language=cpp'],
    cpp_args: cpp_default_args,
    dependencies: omp,
    install: true,
    subdir: 'qbindiff/utils'
)