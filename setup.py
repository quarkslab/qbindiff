import contextlib
import numpy as np
from os.path import normpath
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

main_ns = {}
ver_path = normpath('src/qbindiff/version.py')
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

setup(
    packages=find_packages(
        where="src",
        include=["qbindiff*"],
    ),
    version=main_ns['__version__'],
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
    scripts=["bin/qbindiff"],
)
