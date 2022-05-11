from abc import ABCMeta, abstractmethod
from collections.abc import Iterator
from typing import Any


class GenericGraph(metaclass=ABCMeta):
    """Abstract class representing a generic graph"""

    @abstractmethod
    def items(self) -> Iterator[tuple[Any, Any]]:
        """Return an iterator over the items. Each item is {node_label: node}"""
        raise NotImplementedError()

    @abstractmethod
    def get_node(self, node_label: Any):
        """Returns the node identified by the `node_label`"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def node_labels(self) -> Iterator[Any]:
        """Return an iterator over the node labels"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def nodes(self) -> Iterator[Any]:
        """Return an iterator over the nodes"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def edges(self) -> Iterator[tuple[Any, Any]]:
        """
        Return an iterator over the edges.
        An edge is a pair (node_label_a, node_label_b)
        """
        raise NotImplementedError()
