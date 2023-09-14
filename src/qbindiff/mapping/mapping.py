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

"""Simple mapping interface
"""

from qbindiff.types import Match, ExtendedMapping, Item


class Mapping:
    """
    This class represents an interface to access the result of the matching analysis.
    Its interface is independent of the underlying objects / items manipulated.
    """

    def __init__(
        self, mapping: ExtendedMapping, unmatched_primary: set[Item], unmatched_secondary: set[Item]
    ):
        self._matches = [Match(*x) for x in mapping]
        self._primary_unmatched = unmatched_primary
        self._secondary_unmatched = unmatched_secondary

    def __iter__(self):
        return iter(self._matches)

    @property
    def similarity(self) -> float:
        """
        Sum of similarities of the diff (unbounded value)
        """
        return sum(x.similarity for x in self._matches)

    @property
    def normalized_similarity(self) -> float:
        """
        Normalized similarity of the diff (from 0 to 1)
        """
        return (2 * self.similarity) / (self.nb_item_primary + self.nb_item_secondary)

    @property
    def squares(self) -> float:
        """
        Number of matching squares
        """
        return sum(x.squares for x in self._matches) / 2

    def add_match(
        self,
        item1: Item,
        item2: Item,
        similarity: float = None,
        confidence: float = 0.0,
        squares: int = None,
    ) -> None:
        """
        Add the given match between the two items.

        :param item1: function address in primary
        :param item2: function address in secondary
        :param similarity: similarity metric as float
        :param confidence: confidence in the result (0..1)
        :param squares: Number of squares being made
        :return: None
        """
        self._matches.append(Match(item1, item2, similarity, confidence, squares))

    def remove_match(self, match: Match) -> None:
        """
        Remove the given matching from the matching.

        :param match: Match object to remove from the matching
        :return: None
        """
        self._matches.remove(match)

    @property
    def primary_matched(self) -> set[Item]:
        """
        Set of items matched in primary
        """
        return {x.primary for x in self._matches}

    @property
    def primary_unmatched(self) -> set[Item]:
        """
        Set of items unmatched in primary.
        """
        return self._primary_unmatched

    @property
    def secondary_matched(self) -> set[Item]:
        """
        Set of items matched in the secondary object.
        """
        return {x.secondary for x in self._matches}

    @property
    def secondary_unmatched(self) -> set[Item]:
        """
        Set of items unmatched in the secondary object.
        """
        return self._secondary_unmatched

    @property
    def nb_match(self) -> int:
        """
        Number of matches
        """
        return len(self._matches)

    @property
    def nb_unmatched_primary(self) -> int:
        """
        Number of unmatched items in primary.
        """
        return len(self._primary_unmatched)

    @property
    def nb_unmatched_secondary(self) -> int:
        """
        Number of unmatched items in secondary.
        """
        return len(self._secondary_unmatched)

    @property
    def nb_item_primary(self) -> int:
        """
        Total number of items in primary
        """
        return self.nb_match + self.nb_unmatched_primary

    @property
    def nb_item_secondary(self) -> int:
        """
        Total number of items in secondary.
        """
        return self.nb_match + self.nb_unmatched_secondary

    def match_primary(self, item: Item) -> Match | None:
        """
        Returns the match associated with the given primary item (if any).

        :param item: item to match in primary
        :return: optional match
        """
        for m in self._matches:
            if m.primary == item:
                return m
        return None

    def match_secondary(self, item: Item) -> Match | None:
        """
        Returns the match associated with the given secondary item (if any).

        :param item: item to match in secondary
        :return: optional match
        """
        for m in self._matches:
            if m.secondary == item:
                return m
        return None

    def is_match_primary(self, item: Item) -> bool:
        """
        Returns true if the items in primary did match with an item in secondary.

        :param item: item to match in primary
        :return: whether the item is matched in primary
        """
        return self.match_primary(item) is not None

    def is_match_secondary(self, item: Item) -> bool:
        """
        Returns true if the item in secondary did match with an item in primary.

        :param item: item to match in secondary
        :return: whether the item is matched in secondary
        """
        return self.match_secondary(item) is not None
