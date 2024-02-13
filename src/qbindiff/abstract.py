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

"""Interfaces used by the differ

Contains the common interfaces, defined as abstract classes, that will be used
throught the qbindiff module (the differ, the matcher, the exporters, etc...).
"""

from __future__ import annotations
from abc import ABCMeta, abstractmethod
from collections.abc import Hashable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any
    from qbindiff.types import NodeLabel


class GenericNode(Hashable):
    """
    Abstract class representing a generic node
    """

    @abstractmethod
    def get_label(self) -> NodeLabel:
        """
        Get the label associated to this node

        :returns: The node label associated with this node
        """
        raise NotImplementedError()


class GenericGraph(metaclass=ABCMeta):
    """
    Abstract class representing a generic graph
    """

    @abstractmethod
    def __len__(self) -> int:
        """Number of nodes in the graph"""
        raise NotImplementedError()

    @abstractmethod
    def items(self) -> Iterable[tuple[NodeLabel, GenericNode]]:
        """
        Iterate over the items. Each item is {node_label: node}

        :returns: A :py:class:`Iterable` over the items. Each item is
                  a tuple (node_label, node)
        """
        raise NotImplementedError()

    @abstractmethod
    def get_node(self, node_label: NodeLabel) -> GenericNode:
        """
        Get the node identified by the `node_label`

        :param node_label: the unique identifier of the node
        :returns: The node identified by the label
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def node_labels(self) -> Iterable[NodeLabel]:
        """
        Iterate over the node labels

        :returns: An :py:class:`Iterable` over the node labels
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def nodes(self) -> Iterable[GenericNode]:
        """
        Iterate over the nodes themselves

        :returns: An :py:class:`Iterable` over the nodes
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def edges(self) -> Iterable[tuple[NodeLabel, NodeLabel]]:
        """
        Iterate over the edges. An edge is a pair (node_label_a, node_label_b)

        :returns: An :py:class:`Iterable` over the edges.
        """
        raise NotImplementedError()
