
from setuptools import setup


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
              'qbindiff.loader.backend': 'qbindiff.loader.backend',
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
        'lapjv',
        'networkx'
    ],
)
