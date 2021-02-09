
from setuptools import setup

setup(
    name='qbindiff',
    version='0.1',
    description='QBindiff binary diffing tool based on belief propagation',
    author='Robin David, Elie Mengin',
    author_email='rdavid@quarkslab.com, emengin@quarkslab.com',
    url='https://gitlab.qb/machine_learning/qbindiff',
    packages=['qbindiff',
              'qbindiff.belief',
              'qbindiff.differ',
              'qbindiff.features',
              'qbindiff.loader',
              'qbindiff.loader.backend'],
    package_dir={'qbindiff': 'qbindiff'},
    install_requires=[
        'networkx >= 2.0',
        'click',
        'tqdm',
        'community',
        'scipy',
        'protobuf',
        'pandas',
        'numpy'
    ],
    scripts=['bin/qbindiff']

)

setup(
    name='qbindiff',
    version='0.1',
    description='QBindiff binary diffing tool based on a Network Alignment problem',
    author='Elie Mengin, Robin David',
    author_email='emengin@quarkslab.com, rdavid@quarkslab.com',
    url='https://gitlab.qb/machine_learning/qbindiff',
    package_dir={'qbindiff': 'src'},
    packages={'qbindiff': 'qbindiff',
              'qbindiff.loader': 'qbindiff.loader',
              'qbindiff.features': 'qbindiff.features',
              'qbindiff.matcher': 'qbindiff.matcher',
              'qbindiff.saver': 'qbindiff.saver'},
    scripts=['bin/qbindiff'],
    install_requires=[
        'protobuf',
        'click',
        'tqdm',
        'numpy',
        'scipy',
        'lapjv'
    ],
)
