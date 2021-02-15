
from setuptools import setup
from setuptools import find_packages

setup(
    name='qbindiff',
    version='0.1',
    description='QBindiff binary diffing tool based on a Network Alignment problem',
    author='Elie Mengin, Robin David',
    author_email='emengin@quarkslab.com, rdavid@quarkslab.com',
    url='https://gitlab.qb/machine_learning/qbindiff',
    package_dir=find_packages(),
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
