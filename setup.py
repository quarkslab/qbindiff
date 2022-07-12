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
    )


setup(
    name="qbindiff",
    version="0.2",
    description="QBindiff binary diffing tool based on a Network Alignment problem",
    author=["Elie Mengin", "Robin David", "Riccardo Mori", "Alexis Challande"],
    author_email=[
        "rmori@quarkslab.com",
        "rdavid@quarkslab.com",
        "achallande@quarkslab.com",
    ],
    url="https://gitlab.qb/machine_learning/qbindiff",
    packages=find_packages(
        where="src",
        include=["qbindiff*"],
    ),
    package_dir={"": "src"},
    ext_modules=cythonize_extensions(
        [
            Extension(
                "qbindiff.passes.fast_metrics",
                ["src/qbindiff/passes/fast_metrics.pyx"],
                include_dirs=[np.get_include()],
            ),
            Extension(
                "qbindiff.utils.openmp_helpers",
                ["src/qbindiff/utils/openmp_helpers.pyx"],
                extra_compile_args=["-fopenmp"],
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
        "sklearn",
    ],
    scripts=["bin/qbindiff"],
    extras_require={
        "binexport": ["python-binexport"],
        "quokka": ["quokka"],
        "full": ["python-louvain"],
    },
)
