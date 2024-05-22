python_sources = [files(
    '__init__.py',
    '__main__.py',
    'abstract.py',
    'differ.py',
    'exceptions.py',
    'types.py',
    'version.py',
    'visitor.py',
)]

py.install_sources(
    python_sources,
    subdir: 'qbindiff'
)

subdir('features')
subdir('loader')
subdir('mapping')
subdir('matcher')
subdir('passes')
subdir('utils')