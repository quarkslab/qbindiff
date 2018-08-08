
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
