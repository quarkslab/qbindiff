import json
from pathlib import Path
from collections import namedtuple
from typing import Optional, Generator, Iterator, Iterable

from qbindiff.types import Match, ExtendedMapping, Item


class Mapping:
    """
    This class represents an interface to access the result of the matching analysis.
    """

    def __init__(
        self,
        mapping: ExtendedMapping,
        unmatched_primary: set[Item],
        unmatched_secondary: set[Item],
    ):
        self._matches = [Match(*x) for x in mapping]
        self._primary_unmatched = unmatched_primary
        self._secondary_unmatched = unmatched_secondary

    def __iter__(self):
        return iter(self._matches)

    @property
    def similarity(self) -> float:
        """Global similarity of the diff"""
        return sum(x.similarity for x in self._matches)

    @property
    def normalized_similarity(self) -> float:
        """Global similarity of the diff"""
        return (2 * self.similarity) / (self.nb_item_primary + self.nb_item_secondary)

    @property
    def squares(self) -> float:
        """Number of matchinf squares"""
        return sum(x.squares for x in self._matches) / 2

    def add_match(
        self, item1: Item, item2: Item, similarity: float = None, squares: int = None
    ) -> None:
        """
        Add the given match between the two function addresses
        :param addr_p1: function address in primary
        :param addr_p2: function address in secondary
        :param similarity: similarity metric as float
        :param squares: Number of squares being made
        :return: None
        """
        self._matches.append(Match(item1, item2, similarity, squares))

    def remove_match(self, match: Match) -> None:
        """
        Remove the given matching from the matching
        :param match: Match object to remove from the matching
        :return: None
        """
        self._matches.remove(match)

    @property
    def primary_matched(self) -> set[Item]:
        """
        Provide the set of addresses matched in primary
        :return: set of addresses in primary
        """
        return {x.primary for x in self._matches}

    @property
    def primary_unmatched(self) -> set[Item]:
        """
        Provide the set of addresses matched in primary
        :return: set of addresses in primary
        """
        return self._primary_unmatched

    @property
    def secondary_matched(self) -> set[Item]:
        """
        Provide the set of addresses matched in the secondary binary
        :return: set of addresses in secondary
        """
        return {x.secondary for x in self._matches}

    @property
    def secondary_unmatched(self) -> set[Item]:
        """
        Provide the set of addresses matched in the secondary binary
        :return: set of addresses in secondary
        """
        return self._secondary_unmatched

    @property
    def nb_match(self) -> int:
        """Returns the number of matches"""
        return len(self._matches)

    @property
    def nb_unmatched_primary(self) -> int:
        """Number of unmatched nodes in primary"""
        return len(self._primary_unmatched)

    @property
    def nb_unmatched_secondary(self) -> int:
        """Number of unmatched nodes in secondary"""
        return len(self._secondary_unmatched)

    @property
    def nb_item_primary(self) -> int:
        """Total number of nodes in primary"""
        return self.nb_match + self.nb_unmatched_primary

    @property
    def nb_item_secondary(self) -> int:
        """Total number of nodes in secondary"""
        return self.nb_match + self.nb_unmatched_secondary

    def match_primary(self, item: Item) -> Optional[Match]:
        """Returns the match index associated with the given primary index"""
        for m in self._matches:
            if m.primary == item:
                return m
        return None

    def match_secondary(self, item: Item) -> Optional[Match]:
        """Returns the match index associated with the given secondary index"""
        for m in self._matches:
            if m.secondary == item:
                return m
        return None

    def is_match_primary(self, item: Item) -> bool:
        """Returns true if the address in primary did match with a function"""
        return self.match_primary(item) is not None

    def is_match_secondary(self, item: Item) -> bool:
        """Returns true if the address in secondary did match with a function in primary"""
        return self.match_secondary(item) is not None
