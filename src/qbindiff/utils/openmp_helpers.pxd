# Helpers to safely access OpenMP routines
#
# Those interfaces act as indirections which allows the non-support of OpenMP
# for implementations which have been written for it.
# no-op implementations are provided for the case where OpenMP is not available.
#
# All calls to OpenMP routines should be cimported from this module.

cdef extern from *:
    """
    #ifdef _OPENMP
        #include <omp.h>
        #define QBINDIFF_OPENMP_PARALLELISM_ENABLED 1
    #else
        #define QBINDIFF_OPENMP_PARALLELISM_ENABLED 0
        #define omp_lock_t int
        #define omp_init_lock(l) (void)0
        #define omp_set_lock(l) (void)0
        #define omp_unset_lock(l) (void)0
        #define omp_get_max_threads() 1
    #endif
    """
    bint QBINDIFF_OPENMP_PARALLELISM_ENABLED

    ctypedef struct omp_lock_t:
        pass

    void omp_init_lock(omp_lock_t*) noexcept nogil
    void omp_set_lock(omp_lock_t*) noexcept nogil
    void omp_unset_lock(omp_lock_t*) noexcept nogil
    int omp_get_max_threads() noexcept nogil