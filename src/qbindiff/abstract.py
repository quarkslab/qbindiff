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

from abc import ABCMeta, abstractmethod
from collections.abc import Iterator
from typing import Any


class GenericGraph(metaclass=ABCMeta):
    """
    Abstract class representing a generic graph
    """

    @abstractmethod
    def items(self) -> Iterator[tuple[Any, Any]]:
        """
        Return an iterator over the items. Each item is {node_label: node}
        """
        raise NotImplementedError()

    @abstractmethod
    def get_node(self, node_label: Any):
        """
        Returns the node identified by the `node_label`
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def node_labels(self) -> Iterator[Any]:
        """
        Return an iterator over the node labels
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def nodes(self) -> Iterator[Any]:
        """
        Return an iterator over the nodes
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def edges(self) -> Iterator[tuple[Any, Any]]:
        """
        Return an iterator over the edges.
        An edge is a pair (node_label_a, node_label_b)
        """
        raise NotImplementedError()
