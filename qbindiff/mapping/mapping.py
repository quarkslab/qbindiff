# coding: utf-8
import json
import sqlite3
from typing import Generator, Iterator, Iterable
from pathlib import Path

# Import for types
from qbindiff.types import Tuple, List, Optional, Match
from qbindiff.types import PathLike, Ratio, Idx, Addr, ExtendedMapping
from qbindiff.loader.program import Program



import json
from collections import namedtuple

from qbindiff.types import Addr
from typing import Optional, Set



class Mapping:
    """
    Matching hold all the match data between the two analysed programs
    """
    def __init__(self, mapping: ExtendedMapping, unmatched: Tuple[List[Idx], List[Idx]]):
        self._matches = [Match(*x) for x in mapping]
        self._primary_unmatched = unmatched[0]
        self._secondary_unmatched = unmatched[1]

    def __iter__(self):
        return iter(self._matches)

    @property
    def similarity(self) -> float:
        """ Global similarity of the diff """
        return sum(x.similarity for x in self._matches)

    @property
    def normalized_similarity(self) -> float:
        """ Global similarity of the diff """
        return (2*self.similarity) / (self.nb_item_primary+self.nb_item_secondary)

    @property
    def squares(self) -> float:
        """ Global similarity of the diff """
        return sum(x.squares for x in self._matches) / 2

    def add_match(self, p1: Idx, p2: Idx, similarity: float=None, squares: int=None) -> None:
        """
        Add the given match between the two function addresses
        :param addr_p1: function address in primary
        :param addr_p2: function address in secondary
        :param similarity: similarity metric as float
        :return: None
        """
        self._matches.append(Match(p1, p2, similarity, squares))

    def remove_match(self, match: Match) -> None:
        """
        Remove the given matching from the matching
        :param match: Match object to remove from the matching
        :return: None
        """
        self._matches.remove(match)

    @property
    def primary_matched(self) -> List[Idx]:
        """
        Provide the set of addresses matched in primary
        :return: set of addresses in primary
        """
        return [x.primary for x in self._matches]

    @property
    def primary_unmatched(self) -> List[Idx]:
        """
        Provide the set of addresses matched in primary
        :return: set of addresses in primary
        """
        return self._primary_unmatched

    @property
    def secondary_matched(self) -> List[Idx]:
        """
        Provide the set of addresses matched in the secondary binary
        :return: set of addresses in secondary
        """
        return [x.secondary for x in self._matches]

    @property
    def secondary_unmatched(self) -> List[Idx]:
        """
        Provide the set of addresses matched in the secondary binary
        :return: set of addresses in secondary
        """
        return self._secondary_unmatched

    @property
    def nb_match(self) -> int:
        """ Returns the number of matches """
        return len(self._matches)

    @property
    def nb_unmatched_primary(self) -> int:
        """ Number of unmatched function in the primary program """
        return len(self._primary_unmatched)

    @property
    def nb_unmatched_secondary(self) -> int:
        """ Number of unmatched function in the secondary program """
        return len(self._secondary_unmatched)

    @property
    def nb_item_primary(self) -> int:
        """ Total number of function in primary """
        return self.nb_match + self.nb_unmatched_primary

    @property
    def nb_item_secondary(self) -> int:
        """ Total number of function in secondary """
        return self.nb_match + self.nb_unmatched_secondary

    def match_primary(self, idx: Idx) -> Optional[Match]:
        """ Returns the match index associated with the given primary index """
        for m in self._matches:
            if m.primary == idx:
                return m
        return None

    def match_secondary(self, idx: Idx) -> Optional[Match]:
        """ Returns the match index associated with the given primary index """
        for m in self._matches:
            if m.secondary == idx:
                return m
        return None

    def is_match_primary(self, idx: Idx) -> bool:
        """ Returns true if the address in primary did match with a function """
        return self.match_primary(idx) is not None

    def is_match_secondary(self, idx: Idx) -> bool:
        """ Returns true if the address in secondary did match with a function in primary """
        return self.match_secondary(idx) is not None

    @staticmethod
    def from_file(filename: PathLike) -> 'Mapping':
        with open(filename) as file:
            mapping = json.load(file)
        return Mapping(mapping['matched'], mapping['unmatched'])

    def save(self, filename: PathLike) -> None:
        with open(filename) as file:
            json.dump({'matched': self._matches, 'unmatched': [self._primary_unmatched, self._secondary_unmatched]}, file)





class AddressMapping(Mapping):

    def __init__(self, primary: Iterable, secondary: Iterable,  mapping: ExtendedMapping, unmatched: Tuple[List[Addr], List[Addr]]):
        super().__init__(mapping, unmatched)
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

