from __future__ import absolute_import
import logging

from itertools import chain

# Imports for types
from qbindiff.types import List, Set, Optional, Addr, BeliefMatching
from qbindiff.differ.preprocessing import PreProcessor


class PostProcessor:
    """docstring for PostProcessor"""
    def __init__(self, data: PreProcessor) -> None: # :PreProcessor
        self.primary = data.primary
        self.secondary = data.secondary
        self.primary_features = data.primary_features
        self.secondary_features = data.secondary_features
        self._convert_matching(*data.belief_results)

    def process(self) -> None:
        lonely = self._match_relatives()
        self._match_lonely(lonely)
        self._format_matching()

    def _convert_matching(self, matching: BeliefMatching, objective: float) -> None:
        """
        Converts index matching of Belief propagation into matching of addresses.
        """
        idx1, idx2, weights = zip(*matching)
        adds1 = self.primary_features.index[list(idx1)]
        adds2 = self.secondary_features.index[list(idx1)]
        self.matchindex = dict(zip(adds1, adds2))
        self._similarity = dict(zip(adds1, weights))
        self._objective = objective

    def _match_relatives(self)-> None:
        """
        Matches primary unmatched functions according to their neighborhoods
        Matched if same parents and same children according to the current matchindex as well as same feature_vector
        Lonely functions are recorded for future matchindex
        """
        unmatched = set(self.primary.keys()).difference(self.matchindex)
        lonely = []
        for addr in unmatched:
            candidates = self._get_candidates(addr)
            if candidates is None:
                lonely.append(addr)
                continue
            for candidate in candidates:
                if self._compare_function(addr, candidate):
                    self._objective += 1. + len(self.primary[addr].parents) + len(self.primary[addr].children)
                    self.matchindex.update({addr: candidate})
                    break
        return lonely

    def _match_lonely(self, lonely: List[Addr]) -> None:
        """
        Matches secondary lonely unmatched functions to primary ones
        Matched if same feature_vector
        """
        unmatched = set(self.secondary.keys()).difference(self.matchindex.values())
        lone_candidates = list(filter(lambda addr: self.secondary[addr].is_alone(), unmatched))
        #lone_candidates = [addr for addr in unmatched if self.secondary[addr].is_alone()]
        for addr in lonely:
            for candidate in iter(lone_candidates):
                if self._compare_function(addr, candidate, lone=True):
                    self._objective += 1.
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

    def _format_matching(self) -> None:
        matching = [{"primary": adds1, "secondary": adds2, "similarity": float(self._similarity.get(adds1, 1.))} for adds1, adds2 in self.matchindex.items()]
        unmatched1 = set(self.primary.keys()).difference(self.matchindex.keys())
        unmatched2 = set(self.secondary.keys()).difference(self.matchindex.values())
        matching += [{"primary": adds1, "secondary": None, "similarity": 0.} for adds1 in unmatched1]
        matching += [{"primary": None, "secondary": adds2, "similarity": 0.} for adds2 in unmatched2]
        self.matching = {"score": self._objective, "matching": matching}

