import logging


def is_debug() -> bool:
    """Returns True if the current logging level is set to debug"""
    return logging.root.level <= logging.DEBUG


def iter_csr_matrix(matrix):
    """
    Iter over non-null items in a CSR (Compressed Sparse Row) matrix.
    It returns a generator that, at each iteration, returns the tuple
    (row_index, column_index, value)
    """
    coo_matrix = matrix.tocoo()
    for x, y, v in zip(coo_matrix.row, coo_matrix.col, coo_matrix.data):
        yield (x, y, v)
