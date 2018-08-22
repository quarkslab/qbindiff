import json
from collections import namedtuple

from qbindiff.types import Addr
from typing import Optional

'''
Match represent the matching between two functions and can hold the similarity between the two
'''
Match = namedtuple("Match", "addr_primary addr_secondary similarity")


class Matching:
    '''
    Matching hold all the match data between the two analysed programs
    '''
    def __init__(self, file: str=None):
        self.primary_idx = {}
        self.secondary_idx = {}
        self.unmatched_primary = set()
        self.unmatched_secondary = set()
        self.global_sim = 0

        if file:
            self.load_file(file)

    def __iter__(self):
        for i in self.primary_idx.values():
            yield i

    @property
    def similarity(self) -> float:
        ''' Global similarity of the diff '''
        return self.global_sim

    @similarity.setter
    def similarity(self, value: float) -> None:
        self.global_sim = value

    @property
    def matching(self):
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

    def remove_match(self, match: Match) -> None:
        """
        Remove the given matching from the matching
        :param match: Match object to remove from the matching
        :return: None
        """
        self.primary_idx.pop(match.addr_primary)
        self.secondary_idx.pop(match.addr_secondary)
        del match

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

    def match_secondary(self, addr: Addr) -> Match:
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

    def add_unmatch_primary(self, addr: Addr) -> None:
        """ Add the given address as an unmatched function address of primary """
        self.unmatched_primary.add(addr)

    def add_unmatch_secondary(self, addr: Addr) -> None:
        """ Add the given address as an unmatched function address of secondary """
        self.unmatched_secondary.add(addr)

    def load_file(self, file: str) -> None:
        """ Load the given JSON file which contains matching data """
        with open(file, 'r') as f:
            data = json.load(f)
        self.global_sim = data['similarity']
        for entry in data["matches"]:
            if entry['addr1'] is None:
                self.add_unmatch_secondary(entry['addr2'])
            if entry['addr2'] is None:
                self.add_unmatch_primary(entry['addr1'])
            else:
                addr1, addr2 = entry['addr1'], entry['addr2']
                self.add_match(addr1, addr2, entry['similarity'])

    def write_file(self, out_file: str) -> None:
        """ Output the matching to a json file """
        data = {'similarity': self.similarity, 'matches': []}
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
