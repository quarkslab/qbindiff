from __future__ import absolute_import

import json
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
        with open(filename) as file:
            mapping = json.load(file)
        self.from_mapping(zip(*mapping))

    def load(self, filename: PathLike)
        self.from_file(filename)

    def save(self, filename: PathLike):
        mapping = zip(self.primary_matched, self.secondary_matched, self._similarities, self._nb_squares)
        with open(filename) as file:
            json.dump(mapping, file)

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
    def primary_items(self) -> List[Idx]:
        """
        Provide the list of all indexes in primary
        :return: list of indexes in primary
        WARNING: due to lack of information, this function might be erroneous
        """
        return list(range(max(self._idx)))

    @property
    def secondary_items(self) -> List[Idx]:
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
        return list(set(self.primary_items).difference(self.primary_matched))

    @property
    def secondary_unmatched(self) -> List[Idx]:
        """
        Provide the list of indexes unmatched secondary
        :return: list of indexes in secondary
        """
        return list(set(self.secondary_items).difference(self.secondary_matched))

    @property
    def nb_primary_items(self) -> int:
        """ Total number of function in primary """
        return len(self.primary_items)

    @property
    def nb_secondary_items(self) -> int:
        """ Total number of function in secondary """
        return len(self.secondary_items)

    @property
    def nb_matched(self) -> int:
        """ Returns the number of matches """
        return len(self._idx)

    @property
    def nb_primary_unmatched(self) -> int:
        """ Number of unmatched function in the primary """
        return self.nb_primary_items - self.nb_matched

    @property
    def nb_secondary_unmatched(self) -> int:
        """ Number of unmatched function in the secondary """
        return self.nb_secondary_items - self.nb_matched

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


class FunctionMapping(Mapping)

    def __init__(self, primary: Program, secondary: Program,  mapping: Mapping=None):
        super().__init__(mapping)
        self._primary = primary
        self._secondary = secondary
        self._primary_index = dict(zip(*enumerate(self._primary)))
        self._secondary_index = dict(zip(*enumerate(self._secondary)))

    def primary_match(self, addr: Addr) -> Addr:
        """ Returns the match address associated with the given primary address """
        if addr in self._primary_index.values():
            idx = 
            return self._secondary_index[self._idy[self._idx.indice(idx)]]
        return None

    def secondary_match(self, idx: Idx) -> Idx:
        """ Returns the match index associated with the given secondary index """
        if addr in self._secondary_index.values():
            idx = 
            return self._primary_index[self._idx[self._idy.indice(idx)]]
        return None

    @property
    def primary_items(self) -> List[Addr]:
        """
        Provide the list of all adresses in primary
        :return: list of addresses in primary
        """
        return list(self._primary_index.values())

    @property
    def secondary_items(self) -> List[Addr]:
        """
        Provide the list of all addresses in secondary
        :return: list of addresses in secondary
        """
        return list(self._secondary_index.values())

    @property
    def primary_matched(self) -> List[Addr]:
        """
        Provide the set of addresses matched in primary
        :return: set of addresses in primary
        """
        return [self._primary_index[idx] for idx in super().primary_matched]

    @property
    def secondary_matched(self) -> List[Addr]:
        """
        Provide the set of addresses matched in secondary
        :return: set of addresses in secondary
        """
        return [self._secondary_index[idy] for idy in super().secondary_matched]

    def from_sqlite_file(self, filename: PathLike):
        connect = sqlite3.connect(str(filename))
        cursor = connect.cursor()
        mapping = zip(*cursor.execute('SELECT address1, address2, similarity, squares FROM function').fetchall())
        connect.close()
        self.from_mapping(mapping)

    def save_sqlite(self, filename: PathLike):
        if os.path.exists(str(filename)):
            os.remove(str(filename))
        connect = sqlite3.connect(str(filename))
        cursor = connect.cursor()
        cursor.execute('CREATE TABLE function (address1 INTEGER, address2 INTEGER, similarity REAL, squares INTEGER)')
        cursor.executemany('INSERT INTO function VALUES (?, ?, ?, ?)', zip(self.primary_matched, self.secondary_matched, self._similarities, self._nb_squares))
        connect.commit()
        connect.close()


class BBlockMapping(AddressMapping):
    pass


class FunctionMapping(AddressMapping):
    pass

