from __future__ import absolute_import

import json
from collections import namedtuple

from qbindiff.types import Addr
from typing import Optional, Set

'''
Match represent the matching between two functions and can hold the similarity between the two
'''
Match = namedtuple("Match", "addr_primary addr_secondary similarity")


class Matching:
    """
    Matching hold all the match data between the two analysed programs
    """
    def __init__(self, primary_set: Set[Addr]=None, secondary_set: Set[Addr]=None, file: str=None):
        self.primary_idx = {}
        self.secondary_idx = {}
        self.unmatched_primary = primary_set if primary_set else set()
        self.unmatched_secondary = secondary_set if secondary_set else set()
        self.global_sim = 0

        if file:
            self.load_file(file)

    def __iter__(self):
        for i in self.primary_idx.values():
            yield i

    @property
    def similarity(self) -> float:
        """ Global similarity of the diff """
        return self.global_sim

    @similarity.setter
    def similarity(self, value: float) -> None:
        """ Setter for the global similarity """
        self.global_sim = value

    @property
    def matching(self):
        """
        Provide the matching as a dictionnary from primary addresses to secondary addresses
        :return: dict
        """
        return {x.addr_primary: x.addr_secondary for x in self.primary_idx.values()}

    def add_match(self, addr_p1: Addr, addr_p2: Addr, similarity: float=None) -> None:
        """
        Add the given match between the two function addresses
        :param addr_p1: function address in primary
        :param addr_p2: function address in secondary
        :param similarity: similarity metric as float
        :return: None
        """
        match = Match(addr_p1, addr_p2, float("{0:.2f}".format(similarity)))
        self.primary_idx[addr_p1] = match
        self.secondary_idx[addr_p2] = match
        self.unmatched_primary.remove(addr_p1)
        self.unmatched_secondary.remove(addr_p2)

    def remove_match(self, match: Match) -> None:
        """
        Remove the given matching from the matching
        :param match: Match object to remove from the matching
        :return: None
        """
        self.primary_idx.pop(match.addr_primary)
        self.unmatched_primary.add(match.addr_primary)
        self.secondary_idx.pop(match.addr_secondary)
        self.unmatched_secondary.add(match.addr_secondary)
        del match

    @property
    def primary_address_matched(self) -> Set[Addr]:
        """
        Provide the set of addresses matched in primary
        :return: set of addresses in primary
        """
        return set(self.primary_idx.keys())

    @property
    def secondary_address_matched(self) -> Set[Addr]:
        """
        Provide the set of addresses matched in the secondary binary
        :return: set of addresses in secondary
        """
        return set(self.secondary_idx.keys())

    @property
    def nb_match(self) -> int:
        """ Returns the number of matches """
        return len(self.primary_idx)

    @property
    def nb_unmatched_primary(self) -> int:
        """ Number of unmatched function in the primary program """
        return len(self.unmatched_primary)

    @property
    def nb_unmatched_secondary(self) -> int:
        """ Number of unmatched function in the secondary program """
        return len(self.unmatched_secondary)

    @property
    def nb_function_primary(self) -> int:
        """ Total number of function in primary """
        return self.nb_match + self.nb_unmatched_primary

    @property
    def nb_function_secondary(self) -> int:
        """ Total number of function in secondary """
        return self.nb_match + self.nb_unmatched_secondary

    def match_primary(self, addr: Addr) -> Optional[Match]:
        """ Returns the match object associated with the given primary function address """
        try:
            return self.primary_idx[addr]
        except KeyError:
            return None

    def match_secondary(self, addr: Addr) -> Optional[Match]:
        """ Returns the match object associated with the given secondary function address """
        try:
            return self.secondary_idx[addr]
        except KeyError:
            return None

    def is_match_primary(self, addr: Addr) -> bool:
        """ Returns true if the address in primary did match with a function """
        return addr in self.primary_idx

    def is_match_secondary(self, addr: Addr) -> bool:
        """ Returns true if the address in secondary did match with a function in primary """
        return addr in self.secondary_idx

    def is_unmatch_primary(self, addr: Addr) -> bool:
        """ Returns true if the address is an unmatched function in primary """
        return addr in self.unmatched_primary

    def is_unmatch_secondary(self, addr: Addr) -> bool:
        """ Returns true if the address is an unmatched function in secondary """
        return addr in self.unmatched_secondary

    def load_file(self, file: str) -> None:
        """ Load the given JSON file which contains matching data """
        with open(file, 'r') as f:
            data = json.load(f)
        self.global_sim = data['similarity']
        for entry in data["matches"]:
            addr1, addr2 = entry['addr1'], entry['addr2']
            if addr1 is not None:  # add all addresses in unmatched (will be removed by add_match)
                self.unmatched_secondary.add(entry['addr2'])
            if addr2 is not None:
                self.unmatched_primary.add(entry['addr1'])
            if addr1 and addr2:
                self.add_match(addr1, addr2, entry['similarity'])

    def write_file(self, out_file: str) -> None:
        """ Output the matching to a json file """
        data = {'similarity': float("{0:.2f}".format(self.similarity)), 'matches': []}
        for match in self.primary_idx.values():
            data['matches'].append({'addr1': match.addr_primary,
                                    'addr2': match.addr_secondary,
                                    'similarity': match.similarity})
        for un_p1 in self.unmatched_primary:
            data['matches'].append({'addr1': un_p1, 'addr2': None, 'similarity': 0.0})
        for un_p2 in self.unmatched_secondary:
            data['matches'].append({'addr1': None, 'addr2': un_p2, 'similarity': 0.0})

        with open(out_file, "w") as out:
            json.dump(data, out)
