import contextlib
import numpy as np
from setuptools import setup, Extension, find_packages
from packaging.version import Version


CYTHON_MIN_VERSION = "0.29.24"


def check_cython_version():
    message = (
        f"Please install Cython with a version >= {CYTHON_MIN_VERSION} in order "
        "to build a qbindiff from source."
    )
    try:
        import Cython
    except ModuleNotFoundError as e:
        # Re-raise with more informative error message instead:
        raise ModuleNotFoundError(message) from e

    if Version(Cython.__version__) < Version(CYTHON_MIN_VERSION):
        message += f" The current version of Cython is {Cython.__version__} "
        f"installed in {Cython.__path__}."
        raise ValueError(message)


def cythonize_extensions(extensions):
    """Check that a recent Cython is available and cythonize extensions"""

    check_cython_version()
    from Cython.Build import cythonize

    openmp_supported = True

    n_jobs = 1
    with contextlib.suppress(ImportError):
        import joblib

        n_jobs = joblib.cpu_count()

    debug = True

    return cythonize(
        extensions,
        nthreads=n_jobs,
        compile_time_env={"QBINDIFF_OPENMP_PARALLELISM_ENABLED": openmp_supported},
        compiler_directives={
            "language_level": 3,
            "boundscheck": debug,
            "wraparound": debug,
            "initializedcheck": debug,
            "nonecheck": debug,
            "cdivision": True,
        },
        annotate=debug,
    )


setup(
    name="qbindiff",
    version="0.2",
    description="QBindiff binary diffing tool based on a Network Alignment problem",
    author="Quarkslab",
    author_email="diffing@quarkslab.com",
    url="https://github.com/quarkslab/qbindiff",
    packages=find_packages(
        where="src",
        include=["qbindiff*"],
    ),
    python_requires=">=3.10",
    package_dir={"": "src"},
    ext_modules=cythonize_extensions(
        [
            Extension(
                "qbindiff.matcher.squares",
                ["src/qbindiff/matcher/squares.pyx"],
                include_dirs=[np.get_include()],
                language="c++",
                extra_compile_args=["-fopenmp", "-O3"],
                extra_link_args=["-fopenmp"],
            ),
            Extension(
                "qbindiff.passes.fast_metrics",
                ["src/qbindiff/passes/fast_metrics.pyx"],
                include_dirs=[np.get_include()],
                extra_compile_args=["-fopenmp", "-O3"],
                extra_link_args=["-fopenmp"],
            ),
            Extension(
                "qbindiff.utils.openmp_helpers",
                ["src/qbindiff/utils/openmp_helpers.pyx"],
                extra_compile_args=["-fopenmp", "-O3"],
                extra_link_args=["-fopenmp"],
            ),
        ]
    ),
    install_requires=[
        "click",
        "tqdm",
        "numpy",
        "scipy",
        "lapjv",
        "networkx",
        "capstone",
        "datasketch",
        "scikit-learn",
        "python-louvain",
        "enum_tools",
        "python-bindiff",
        "tox",
    ],
    scripts=["bin/qbindiff"],
    extras_require={
        "binexport": ["python-binexport"],
        "quokka": ["quokka-project"],
        "doc": [
            "sphinx>=7.2.0",
            "sphinx-design",
            "sphinx-rtd-theme",
            "enum-tools[sphinx]",
            "sphinx_autodoc_typehints",
        ],
        "community": ["python-louvain"],
    },
)
