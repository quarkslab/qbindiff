python_sources = [files(
    '__init__.py',
    'base.py',
    'metrics.py',
    'passes.py',
    'utils.py',
)]

py.install_sources(
    python_sources,
    subdir: 'qbindiff/passes'
)

py.extension_module(
    'fast_metrics',
    cython_gen_cpp.process('fast_metrics.pyx'),
    override_options: ['cython_language=cpp'],
    cpp_args: cpp_default_args,
    dependencies: [np_dep, omp],
    install: true,
    subdir: 'qbindiff/passes'
)