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

"""Program
"""

from __future__ import annotations
from collections.abc import MutableMapping
from typing import TYPE_CHECKING
from pathlib import Path

from qbindiff.abstract import GenericGraph
from qbindiff.loader import Function
from qbindiff.loader.types import LoaderType

if TYPE_CHECKING:
    import networkx
    from networkx.classes.reportviews import OutEdgeView
    from collections.abc import Callable, Iterator
    from qbindiff.loader import Structure
    from qbindiff.loader.backend.abstract import AbstractProgramBackend
    from qbindiff.types import Addr


class Program(MutableMapping, GenericGraph):
    """
    Program class that shadows the underlying program backend used.

    It is a :py:class:`MutableMapping`, where keys are function addresses and
    values are :py:class:`Function` objects.

    :param path: Path to the main file to load (depends on the underlying backend)
    :param kwargs: Valid kwargs are:
        * loader: LoaderType | None for the loader type
        * backend: object, object instance implementing the appropriate interface

    The node label is the function address, the node itself is the :py:class:`Function` object
    """

    def __init__(self, path: Path | str, *args, **kwargs):
        super().__init__()
        path = Path(path)
        loader = kwargs.pop("loader") if "loader" in kwargs else None

        # if a backend instance is directly provided use it
        if loader is None and (backend := kwargs.get("backend")) is not None:
            self._backend = backend  # Load directly from instanciated backend

        # Try to infer it
        if loader is None:
            if path.suffix.casefold() == ".Quokka".casefold():
                loader = LoaderType.quokka
            elif path.suffix.casefold() == ".BinExport".casefold():
                loader = LoaderType.binexport

        # Match the resulting loader
        if loader == LoaderType.ida:
            from qbindiff.loader.backend.ida import ProgramBackendIDA
            self._backend = ProgramBackendIDA(*args, **kwargs)

        elif loader == LoaderType.binexport:
            from qbindiff.loader.backend.binexport import ProgramBackendBinExport
            self._backend = ProgramBackendBinExport(str(path), *args, **kwargs)

        elif loader == LoaderType.quokka:
            from qbindiff.loader.backend.quokka import ProgramBackendQuokka
            self._backend = ProgramBackendQuokka(str(path), *args, **kwargs)

        else:
            raise NotImplementedError(f"Loader: {loader} not implemented")

        self._filter = lambda x: True
        self._functions: dict[Addr, Function] = {}  # underlying dictionary containing the functions
        self._load_functions()

    @staticmethod
    def from_binexport(file_path: str, arch: str | None = None) -> Program:
        """
        Load the Program using the binexport backend

        :param file_path: File path to the binexport file
        :param arch: Architecture to pass to the capstone disassembler. This is
                     useful when the binexport'ed architecture is not enough to
                     correctly disassemble the binary (for example with arm
                     thumb2 or some mips modes).
        :return: Program instance
        """

        return Program(LoaderType.binexport, file_path, arch=arch)

    @staticmethod
    def from_quokka(file_path: str, exec_path: str) -> Program:
        """
        Load the Program using the Quokka backend.

        :param file_path: File path to the binexport file
        :param exec_path: Path of the raw binary
        :return: Program instance
        """

        return Program(LoaderType.quokka, file_path, exec_path=exec_path)

    @staticmethod
    def from_ida() -> Program:
        """
        Load the program using the IDA backend

        :return: Program instance
        """

        return Program(LoaderType.ida)

    @staticmethod
    def from_backend(backend: AbstractProgramBackend) -> Program:
        """
        Load the Program from an instanciated program backend object
        """

        return Program(None, backend=backend)

    def __repr__(self) -> str:
        return "<Program:%s>" % self.name

    def __iter__(self) -> Iterator[Function]:
        """
        Iterate over all functions located in the program, using the filter registered.

        :return: Iterator of all the functions
        """

        yield from self._functions.values()

    def __len__(self) -> int:
        return len(self._functions)

    def __getitem__(self, key):
        return self._functions.__getitem__(key)

    def __setitem__(self, key, value):
        self._functions.__setitem__(key, value)

    def __delitem__(self, key):
        self._functions.__delitem__(key)

    def _load_functions(self) -> None:
        """Load the functions from the backend"""

        for function in map(Function.from_backend, self._backend.functions):
            self[function.addr] = function

    def items(self) -> Iterator[tuple[Addr, Function]]:  # type: ignore[override]
        """
        Iterate over the items. Each item is {address: :py:class:`Function`}

        :returns: A :py:class:`Iterator` over the functions. Each element
                  is a tuple (function_addr, function_obj)
        """

        # yield function only if filter agree to keep it
        yield from filter(lambda i: self._filter(i[0]), self._functions.items())

    def get_node(self, node_label: Addr) -> Function:
        """
        Get the function identified by the address ``node_label``

        :param node_label: the address of the function that will be returned
        :returns: the function identified by its address
        """

        return self[node_label]

    @property
    def node_labels(self) -> Iterator[Addr]:
        """
        Iterate over the functions' address

        :returns: An :py:class:`Iterator` over the functions' address
        """

        yield from filter(self._filter, self._functions.keys())

    @property
    def nodes(self) -> Iterator[Function]:
        """
        Iterate over the functions

        :returns: An :py:class:`Iterator` over the functions
        """

        yield from self.__iter__()

    @property
    def edges(self) -> OutEdgeView[tuple[Addr, Addr]]:
        """
        Iterate over the edges. An edge is a pair (addr_a, addr_b)

        :returns: An :py:class:`OutEdgeView` over the edges.
        """

        return self.callgraph.edges

    @property
    def name(self) -> str:
        """
        Returns the name of the program as defined by the backend
        """

        return self._backend.name

    @property
    def structures(self) -> list[Structure]:
        """
        Returns the list of structures defined in program
        """

        return self._backend.structures

    @property
    def exec_path(self) -> str | None:
        """
        The executable path if it has been specified, None otherwise
        """

        return self._backend.exec_path

    def set_function_filter(self, func: Callable[[Addr], bool]) -> None:
        """
        Filter out some functions, to ignore them in later processing.

        .. warning: The filter only apply for __iter__, items functions and callgraph
                    property.
                    Accessing functions through the dictionary does not apply the filter

        :param func: function take the function address (the node label) and returns
                     whether or not to keep it.
        """

        self._filter = func

    @property
    def callgraph(self) -> networkx.DiGraph:
        """
        The function callgraph with a Networkx DiGraph
        """

        cg = self._backend.callgraph
        funcs = list(self)  # functions already filtered
        return cg.subgraph([x.addr for x in funcs])

    def get_function(self, name: str) -> Function:
        """
        Returns the function by its name

        :param name: name of the function
        :return: the function
        """

        return self[self._backend.fun_names[name]]

    def follow_through(self, to_remove: Addr, target: Addr) -> None:
        """
        Replace node `to_remove` with a follow-through edge from every parent of the
        node with the node `target`.

        Example : ``{ parents } -> (to_remove) -> (target)``

        ``--> { parents } -> (target)``

        :param to_remove: node to remove
        :param target: targe node
        :return: None
        """

        func = self[to_remove]
        self.pop(to_remove)
        for p_addr in list(func.parents):
            # Remove edges
            self[p_addr].children.remove(to_remove)
            func.parents.remove(p_addr)
            self._backend.callgraph.remove_edge(p_addr, to_remove)
            # Add follow-through edge
            self[p_addr].children.add(target)
            self[target].parents.add(p_addr)
            self._backend.callgraph.add_edge(p_addr, target)
        for c_addr in list(func.children):
            # Remove edges
            func.children.remove(c_addr)
            self[c_addr].parents.remove(to_remove)
            self._backend.callgraph.remove_edge(to_remove, c_addr)
        self._backend.callgraph.remove_node(to_remove)

    def remove_function(self, to_remove: Addr) -> None:
        """
        Remove the node ``to_remove`` from the Call Graph of the program.

        **WARNING**: The follow-through edges from the parents to the children are not
        added. Example :

        ``{ parents } -> (to_remove) -> { children }``

        ``--> { parents }                   { children }``

        :param to_remove: function_to_remove
        :return: None
        """

        func = self[to_remove]
        self.pop(to_remove)
        for p_addr in list(func.parents):
            # Remove edges
            self[p_addr].children.remove(to_remove)
            func.parents.remove(p_addr)
            self._backend.callgraph.remove_edge(p_addr, to_remove)
        for c_addr in list(func.children):
            # Remove edges
            func.children.remove(c_addr)
            self[c_addr].parents.remove(to_remove)
            self._backend.callgraph.remove_edge(to_remove, c_addr)
        self._backend.callgraph.remove_node(to_remove)
