# Copyright 2023 Quarkslab
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Collection of utilities

Collection of utilities used internally.
"""

from __future__ import annotations
from functools import cache
from collections.abc import Iterator
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator
    from typing import Any
    from qbindiff.types import SparseMatrix


def is_debug() -> bool:
    """Returns True if the current logging level is set to debug"""
    return logging.root.level <= logging.DEBUG


def iter_csr_matrix(matrix: SparseMatrix) -> Generator[tuple[int, int, Any], None, None]:
    """
    Iterate over non-null items in a CSR (Compressed Sparse Row) matrix.
    It returns a generator that, at each iteration, returns the tuple
    (row_index, column_index, value)

    :param matrix: CSR matrix
    :return: generator (row_idx, column_idx, val)
    """

    coo_matrix = matrix.tocoo()
    for x, y, v in zip(coo_matrix.row, coo_matrix.col, coo_matrix.data):
        yield (x, y, v)


@cache
def log_once(level: int, message: str) -> None:
    """
    Log a message with the corresponding level only once.

    :param level: The severity level of the logging
    :param message: The message to log
    """

    logging.log(level, message)


def wrapper_iter(opt_iter: Iterator[Any] | Any, default_value: Any | None = 1000) -> Iterator[Any]:
    """
    Utility function that wraps an iterator over an optional iterator.
    If the argument provided is not an iterator the function will return an iterator
    over a list containing only the default_value param.

    :param opt_iter: The optional iterator.
    :param default_value: The default value that will be returned by the iterator if
                          the argument is not an iterator
    :returns: The untouched argument if it is an iterator, otherwise an iterator
              over the list [default_value]
    """

    if isinstance(opt_iter, Iterator):
        return opt_iter
    else:
        return iter([default_value])
