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

from __future__ import annotations
import csv, logging
from typing import TYPE_CHECKING

from qbindiff.types import Match

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Callable
    from qbindiff.types import ExtendedMapping, Node

    ExtraAttrsType: TypeAlias = str | tuple[str, Callable[[Node], Any]]


class Mapping:
    """
    This class represents an interface to access the result of the matching analysis.
    Its interface is independent of the underlying :py:obj:`Node`s manipulated.
    """

    def __init__(
        self, mapping: ExtendedMapping, unmatched_primary: set[Node], unmatched_secondary: set[Node]
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
        return (2 * self.similarity) / (self.nb_node_primary + self.nb_node_secondary)

    @property
    def squares(self) -> float:
        """
        Number of matching squares
        """
        return sum(x.squares for x in self._matches) / 2

    def add_match(
        self,
        node1: Node,
        node2: Node,
        similarity: float = None,
        confidence: float = 0.0,
        squares: int = None,
    ) -> None:
        """
        Add the given match between the two nodes.

        :param node1: node in primary
        :param node2: node in secondary
        :param similarity: similarity metric as float
        :param confidence: confidence in the result (0..1)
        :param squares: Number of squares being made
        :return: None
        """
        self._matches.append(Match(node1, node2, similarity, confidence, squares))

    def remove_match(self, match: Match) -> None:
        """
        Remove the given matching from the matching.

        :param match: Match object to remove from the matching
        :return: None
        """
        self._matches.remove(match)

    @property
    def primary_matched(self) -> set[Node]:
        """
        Set of nodes matched in primary
        """
        return {x.primary for x in self._matches}

    @property
    def primary_unmatched(self) -> set[Node]:
        """
        Set of nodes unmatched in primary.
        """
        return self._primary_unmatched

    @property
    def secondary_matched(self) -> set[Node]:
        """
        Set of nodes matched in the secondary object.
        """
        return {x.secondary for x in self._matches}

    @property
    def secondary_unmatched(self) -> set[Node]:
        """
        Set of nodes unmatched in the secondary object.
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
        Number of unmatched nodes in primary.
        """
        return len(self._primary_unmatched)

    @property
    def nb_unmatched_secondary(self) -> int:
        """
        Number of unmatched nodes in secondary.
        """
        return len(self._secondary_unmatched)

    @property
    def nb_nodes_primary(self) -> int:
        """
        Total number of nodes in primary
        """
        return self.nb_match + self.nb_unmatched_primary

    @property
    def nb_nodes_secondary(self) -> int:
        """
        Total number of nodes in secondary.
        """
        return self.nb_match + self.nb_unmatched_secondary

    def match_primary(self, node: Node) -> Match | None:
        """
        Returns the match associated with the given primary node (if any).

        :param node: node to match in primary
        :return: optional match
        """
        for m in self._matches:
            if m.primary == node:
                return m
        return None

    def match_secondary(self, node: Node) -> Match | None:
        """
        Returns the match associated with the given secondary node (if any).

        :param node: node to match in secondary
        :return: optional match
        """
        for m in self._matches:
            if m.secondary == node:
                return m
        return None

    def is_match_primary(self, node: Node) -> bool:
        """
        Returns true if the node in primary has been matched with a node in secondary.

        :param node: ndoe to match in primary
        :returns: whether the node has been matched
        """
        return self.match_primary(node) is not None

    def is_match_secondary(self, node: Node) -> bool:
        """
        Returns true if the node in secondary has been matched with a node in primary.

        :param node: ndoe to match in secondary
        :returns: whether the node has been matched
        """
        return self.match_secondary(node) is not None

    def to_csv(self, path: Path | str, *extra_attrs: *ExtraAttrsType) -> None:
        """
        Write the mapping into a csv file.
        Additional attributes of the nodes to put in the csv can be optionally specified.

        For example:
        .. code-block:: python
            :linenos:

            # Adding the attribute "primary_addr" and "secondary_addr". The value will be obtained
            # by accessing `function.addr`
            mapping.to_csv("result.csv", "addr")

            # Adding the attributes name and type. This will add the fields "primary_name",
            # "secondary_name", "primary_type", "secondary_type"
            mapping.to_csv("result.csv", ("name", lambda f: f.name.upper()), "type")

        :param path: The file path of the csv file to write
        :param extra_attrs: Additional attributes to put in the csv. Each attribute is either a
                            tuple (attribute_name, attribute_function) or a string *attribute_name*
        """

        # Check the path
        if isinstance(path, str):
            path = Path(str)
        if path.exists() and not path.is_file():
            raise ValueError(f"path `{path}` already exists and is not a file.")
        if path.exists():
            logging.info(f"Overwriting file {path}")

        # Extract the optional extra attributes
        attrs_name = []
        attrs_func = []
        for extra_attr in extra_attrs:
            match extra_attr:
                case str(name):
                    attrs_name.append(f"primary_{name}")
                    attrs_name.append(f"secondary_{name}")
                    attrs_func.append(lambda f: getattr(f, name))
                case (name, func):
                    attrs_name.append(f"primary_{name}")
                    attrs_name.append(f"secondary_{name}")
                    attrs_func.append(func)

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ("primary_node", "secondary_node", "similarity", "confidence", *attrs_name)
            )
            for match in self._matches:
                # Get the extra attributes values
                extra_values = []
                for func in attrs_func:
                    extra_values.append(func(match.primary))
                    extra_values.append(func(match.secondary))

                writer.writerow(
                    (
                        match.primary.get_label(),
                        match.secondary.get_label(),
                        match.similarity,
                        match.confidence,
                        *extra_values,
                    )
                )
