from __future__ import annotations
import networkx, gc, sys
from collections.abc import Mapping
from typing import Set

from qbindiff.loader import BasicBlock
from qbindiff.loader.types import LoaderType, FunctionType
from qbindiff.types import Addr


class Function(Mapping[Addr, BasicBlock]):
    """
    Representation of a binary function.
    This class is a dict of basic block addreses to the basic block.
    It lazily loads all the basic blocks when iterating through them or even accessing
    one of them and it unloads all of them after the iteration has ended.

    To keep a reference to the basic blocks the `with` statement can be used, for example:
    ```
    # func: Function
    with func:  # Loading all the basic blocks
        for bb_addr, bb in func.items():  # Blocks are already loaded
            pass
        # The blocks are still loaded
        for bb_addr, bb in func.items():
            pass
    # here the blocks have been unloaded
    ```
    """

    def __init__(self, loader: LoaderType | None, /, *args, **kwargs):
        super(Function, self).__init__()

        # The basic blocks are lazily loaded
        self._basic_blocks = None
        self._enable_unloading = True

        self._backend = None
        if loader is None and (backend := kwargs.get("backend")) is not None:
            self._backend = backend  # Load directly from instanciated backend
        elif loader == LoaderType.binexport:
            self.load_binexport(*args, **kwargs)
        elif loader == LoaderType.ida:
            self.load_ida(*args, **kwargs)
        elif loader == LoaderType.quokka:
            self.load_quokka(*args, **kwargs)
        else:
            raise NotImplementedError("Loader: %s not implemented" % loader)

    def load_binexport(self, *args, **kwargs):
        from qbindiff.loader.backend.binexport import FunctionBackendBinExport

        self._backend = FunctionBackendBinExport(*args, **kwargs)

    def load_ida(self, addr):
        from qbindiff.loader.backend.ida import FunctionBackendIDA

        self._backend = FunctionBackendIDA(self, addr)

    def load_quokka(self, *args, **kwargs):
        from qbindiff.loader.backend.quokka import FunctionBackendQuokka

        self._backend = FunctionBackendQuokka(*args, **kwargs)

    @staticmethod
    def from_backend(backend: AbstractFunctionBackend) -> Function:
        """Load the Function from an instanciated function backend object"""
        return Function(None, backend=backend)

    def __hash__(self):
        return hash(self.addr)

    def __enter__(self):
        """Preload basic blocks and don't deallocate them until __exit__ is called"""
        self._enable_unloading = False
        self._preload()

    def __exit__(self, exc_type, exc_value, traceback):
        """Deallocate all the basic blocks"""
        self._enable_unloading = True
        self._unload()

    def __getitem__(self, key: Addr) -> BasicBlock:
        if self._basic_blocks is not None:
            return self._basic_blocks[key]

        self._preload()
        bb = self._basic_blocks[key]
        self._unload()
        return bb

    def __iter__(self):
        """Iterate over basic blocks, not addresses"""
        if self._basic_blocks is not None:
            yield from self._basic_blocks.values()
        else:
            self._preload()
            yield from self._basic_blocks.values()
            self._unload()

    def __len__(self):
        if self._basic_blocks is not None:
            return len(self._basic_blocks)

        self._preload()
        size = len(self._basic_blocks)
        self._unload()
        return size

    def items(self):
        if self._basic_blocks is not None:
            yield from self._basic_blocks.items()
        else:
            self._preload()
            yield from self._basic_blocks.items()
            self._unload()

    def _preload(self) -> None:
        """Load in memory all the basic blocks"""
        self._basic_blocks = {}
        for bb in map(BasicBlock.from_backend, self._backend.basic_blocks):
            self._basic_blocks[bb.addr] = bb

    def _unload(self) -> None:
        """Unload from memory all the basic blocks"""
        if self._enable_unloading:
            self._basic_blocks = None
            self._backend.unload_blocks()

    @property
    def edges(self):
        return list(self.flowgraph.edges)

    @property
    def addr(self) -> Addr:
        """
        Address of the function
        :return: addr
        """
        return self._backend.addr

    @property
    def flowgraph(self) -> networkx.DiGraph:
        """
        Gives the networkx DiGraph of the function. This is used to perform networkx
        based algorithm.
        :return: directed graph of the function
        """
        return self._backend.graph

    @property
    def parents(self) -> Set[Addr]:
        """
        Set of function parents in the call graph. Thus
        functions that calls this function
        :return: caller functions
        """
        return self._backend.parents

    @property
    def children(self) -> Set[Addr]:
        """
        Set of functions called by this function in the
        call graph.
        :return: callee functions
        """
        return self._backend.children

    @property
    def type(self) -> FunctionType:
        """
        Returns the type of the instruction (as defined by IDA)
        :return: function type
        """
        return self._backend.type

    @type.setter
    def type(self, value) -> None:
        """Set the type value"""
        self._backend.type = value

    def is_library(self) -> bool:
        """
        Returns whether or not this function is a library function.
        A library function is either a thunk function or it has been identified as part
        of an external library. It is not an imported function.
        :return: bool
        """
        return self.type == FunctionType.library

    def is_import(self) -> bool:
        """
        Returns whether or not this function is an import function.
        (Thus not having content)
        :return: bool
        """
        return self.type in (FunctionType.imported, FunctionType.extern)

    def is_thunk(self) -> bool:
        """
        Returns whether or not this function is a thunk function.
        :return: bool
        """
        return self.type == FunctionType.thunk

    def is_alone(self):
        """
        Returns whether or not the function have neither
        caller nor callee.
        :return: bool
        """
        return not (self.children or self.parents)

    def __repr__(self):
        return "<Function: 0x%x>" % self.addr

    @property
    def name(self):
        return self._backend.name

    @name.setter
    def name(self, name):
        self._backend.name = name
