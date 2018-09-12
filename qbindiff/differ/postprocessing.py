from __future__ import absolute_import
import logging

from itertools import chain

# Imports for types
from qbindiff.loader.program import Program
from qbindiff.types import Set, Optional, Addr, DataFrame, PureMatching


class Postprocessor:
    """docstring for PostProcessor"""
    def __init__(self, primary: Program, secondary: Program, primary_features: DataFrame, secondary_features: DataFrame, matchindex: PureMatching) -> None:
        self.primary = primary
        self.secondary = secondary
        self.primary_features = primary_features
        self.secondary_features = secondary_features
        self.matchindex = matchindex
        self.add_objective = 0

    def match_relatives(self)-> None:
        """
        Matches primary unmatched functions according to their neighborhoods
        Matched if same parents and same children according to the current matchindex as well as same feature_vector
        Lonely functions are recorded for future matchindex
        """
        unmatched = set(self.primary.keys()).difference(self.matchindex)
        self.lonely = []
        for addr in unmatched:
            candidates = self._get_candidates(addr)
            if candidates is None:
                self.lonely.append(addr)
                continue
            for candidate in candidates:
                if self._compare_function(addr, candidate):
                    self.add_objective += 1. + len(self.primary[addr].parents) + len(self.primary[addr].children)
                    self.matchindex.update({addr: candidate})
                    break

    def match_lonely(self) -> None:
        """
        Matches secondary lonely unmatched functions to primary ones
        Matched if same feature_vector
        """
        unmatched = set(self.secondary.keys()).difference(self.matchindex.values())
        lone_candidates = list(filter(lambda addr: self.secondary[addr].is_alone(), unmatched))
        #lone_candidates = [addr for addr in unmatched if self.secondary[addr].is_alone()]
        for addr in self.lonely:
            for candidate in lone_candidates:
                if self._compare_function(addr, candidate, lone=True):
                    self.add_objective += 1.
                    self.matchindex.update({addr: candidate})
                    lone_candidates.remove(candidate)
                    break

    def _get_candidates(self, addr: Addr) -> Optional[Set[Addr]]:
        """
        Extracts the candidate set of the function "address"
        Intersects the children sets of all secondary functions matched with parents of "address" in primay
        Do the same for parents of functions coupled with children of "address"
        Return the union of both sets
        """
        if self.primary[addr].is_alone():
            return None
        candidates = set()
        if self.primary[addr].parents:
            parentcandidates = [self.secondary[parentmatch].children for parentmatch in map(self.matchindex.get, self.primary[addr].parents) if parentmatch]
            candidates.update(*chain(parentcandidates))
        if self.primary[addr].children:
            childrencandidates = [self.secondary[childrenmatch].parents for childrenmatch in map(self.matchindex.get, self.primary[addr].children) if childrenmatch]
            candidates.update(*chain(childrencandidates))
        return candidates

    def _compare_function(self, addr1: Addr, addr2: Addr, lone: bool=False) -> bool:
        """
        True if adds1 and adds2 have the same parents and the same children according
        to the current matchindex as well as same feature_vector
        """
        if (self.primary_features.loc[addr1] != self.secondary_features.loc[addr2]).any():
            return False
        if lone:
            return True
        # RMQ: Is None expected in the set (cc. elie)?
        if self.secondary[addr2].parents != {self.matchindex.get(x) for x in self.primary[addr1].parents}:
            return False
        if self.secondary[addr2].children != {self.matchindex.get(x) for x in self.primary[addr1].children}:
            return False
        return True

