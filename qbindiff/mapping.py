from __future__ import absolute_import

import sqlite3
from pathlib import Path

from typing import Tuple, List, Optional
from qbindiff.types import PathLike, Mapping, Idx, Addr



class Mapping:
    """
    Matching hold all the match data between the two analysed programs
    """
    def __init__(self, mapping: Optional[Mapping]=None):
        self._idx = []
        self._idy = []
        self._similarities = []
        self._nb_squares = []
        if mapping:
            self.from_mapping(mapping)

    def __iter__(self) -> Tuple[List[Idx], List[Idx]]:
        return self.primary_match, self.secondary_matched

    def from_mapping(self, mapping: Mapping):
        idx, idy, _similarities, nb_squares = mapping
        self._idx = list(idx)
        self._idy = list(idy)
        self._similarities = list(similarities)
        self._nb_squares = list(nb_squares)

    def from_file(self, filename: PathLike):
        connect = sqlite3.connect(str(filename))
        cursor = connect.cursor()
        mapping = zip(*cursor.execute('SELECT address1, address2, similarity, squares FROM function').fetchall())
        connect.close()
        self.from_mapping(mapping)

    def load(self, filename: PathLike)
        self.from_file(filename)

    def save(self, filename: PathLike):
        if filename is None:
            filename = '{}_vs_{}.qbindiff'.format(self.primary.name, self.secondary.name)
        if os.path.exists(str(filename)):
            os.remove(str(filename))
        connect = sqlite3.connect(str(filename))
        cursor = connect.cursor()
        cursor.execute('DROP TABLE IF EXISTS function')
        cursor.execute('CREATE TABLE function (address1 INTEGER, address2 INTEGER, similarity REAL, squares INTEGER)')
        cursor.executemany('INSERT INTO function VALUES (?, ?, ?, ?)', zip(self.primary_matched, self.secondary_matched, self._similarities, self._nb_squares))
        connect.commit()
        connect.close()

    def primary_match(self, idx: Idx) -> Idx:
        """ Returns the match index associated with the given primary index """
        if idx in self._idx:
            return self._idy[self._idx.indice(idx)]
        return None

    def secondary_match(self, idx: Idx) -> Idx:
        """ Returns the match index associated with the given secondary index """
        if idx in self._idy:
            return self._idx[self._idy.indice(idx)]
        return None

    @property
    def primary_functions(self) -> List[Idx]:
        """
        Provide the list of all indexes in primary
        :return: list of indexes in primary
        WARNING: due to lack of information, this function might be erroneous
        """
        return list(range(max(self._idx)))

    @property
    def secondary_functions(self) -> List[Idx]:
        """
        Provide the list of all indexes secondary
        :return: list of indexes in secondary
        WARNING: due to lack of information, this function might be erroneous
        """
        return list(range(max(self._idy)))

    @property
    def primary_matched(self) -> List[Idx]:
        """
        Provide the list of indexes matched in primary
        :return: list of indexes in primary
        """
        return self._idx

    @property
    def secondary_matched(self) -> List[Idx]:
        """
        Provide the list of indexes matched secondary
        :return: list of indexes in secondary
        """
        return self._idy

    @property
    def primary_unmatched(self) -> List[Idx]:
        """
        Provide the list of indexes unmatched primary
        :return: list of indexes in primary
        """
        return list(set(self.primary_functions).difference(self.primary_matched))

    @property
    def secondary_unmatched(self) -> List[Idx]:
        """
        Provide the list of indexes unmatched secondary
        :return: list of indexes in secondary
        """
        return list(set(self.secondary_functions).difference(self.secondary_matched))

    @property
    def nb_primary_functions(self) -> int:
        """ Total number of function in primary """
        return len(self.primary_functions)

    @property
    def nb_secondary_functions(self) -> int:
        """ Total number of function in secondary """
        return len(self.secondary_functions)

    @property
    def nb_matched(self) -> int:
        """ Returns the number of matches """
        return len(self._idx)

    @property
    def nb_primary_unmatched(self) -> int:
        """ Number of unmatched function in the primary """
        return self.nb_primary_functions - self.nb_matched

    @property
    def nb_secondary_unmatched(self) -> int:
        """ Number of unmatched function in the secondary """
        return self.nb_secondary_functions - self.nb_matched

    @property
    def similarity(self) -> float:
        """ Global similairty score of the mapping """
        return sum(self._similarities)

    @property    
    def nb_squares(self) -> int:
        """ Global square number of the mapping """
        return sum(self._nb_squares) / 2

    @property    
    def score(self, tradeoff: float=.5) -> float:
        """ Global network alignment score of the mapping """
        return tradeoff * self.similarity + (1 - tradeoff) * self.nb_squares

    def display_statistics(self, mapping: Optional[Mapping]=None):
        output = 'Score: {:.4f} | Similarity: {:.4f} | '\
                 'Squares: {:.0f} | Nb matches: {}\n'.format(self.score, self.similarity, self.nb_squares, self.nb_matched)
        return output



class FunctionMapping(Mapping)

    def __init__(self, primary: Program, secondary: Program,  mapping: Mapping=None):
        super().__init__(mapping)
        self._primary = primary
        self._secondary = secondary
        self._primary_index = dict(zip(*enumerate(self._primary)))
        self._secondary_index = dict(zip(*enumerate(self._secondary)))

    @property
    def primary_functions(self) -> List[Addr]:
        """ Total number of function in primary """
        return list(self._primary_index.values())

    @property
    def secondary_functions(self) -> List[Addr]:
        """ Total number of function in secondary """
        return list(self._secondary_index.values())

    @property
    def primary_matched(self) -> List[Addr]:
        """
        Provide the set of indexes matched in primary
        :return: set of indexes in primary
        """
        return [self._primary_index[idx] for idx in self._idx]

    @property
    def secondary_matched(self) -> List[Addr]:
        """
        Provide the set of indexes matched in the secondary binary
        :return: set of indexes in secondary
        """
        return [self._secondary_index[idy] for idy in self._idy]

    def display_statistics(self, mapping: Optional[Mapping]=None):
        output = super().display_statistics(mapping)
        output += 'Node cover:  {:.3f}% / {:.3f}% | '\
                  'Edge cover:  {:.3f}% / {:.3f}%\n'.format(100 * self.nb_matched / self.nb_primary_functions,
                                                            100 * self.nb_matched / self.nb_secondary_functions,
                                                            100 * self.nb_squares / self._primary.callgraph.sum(),
                                                            100 * self.nb_squares / self._secondary.callgraph.sum())
        return output


class BasicBlockMapping(AddressMapping):
    pass


class FunctionMapping(AddressMapping):
    pass

