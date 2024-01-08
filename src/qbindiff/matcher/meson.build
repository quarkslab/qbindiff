python_sources = [files(
    '__init__.py',
    'belief_propagation.py',
    'matcher.py',
)]

py.install_sources(
    python_sources,
    subdir: 'qbindiff/matcher'
)

py.extension_module(
    'squares',
    cython_gen_cpp.process('squares.pyx'),
    override_options: ['cython_language=cpp'],
    cpp_args: cpp_default_args,
    dependencies: [np_dep, omp],
    install: true,
    subdir: 'qbindiff/matcher'
)