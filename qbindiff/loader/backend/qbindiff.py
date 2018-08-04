import json
import os.path
from pathlib import Path
import logging
import networkx

from qbindiff.loader.function import Function
from qbindiff.loader.types import LoaderType, OperandType
from qbindiff.loader.instruction import Instruction
from qbindiff.loader.operand import Operand

map_table = {'void': OperandType.void,
             'reg': OperandType.register,
             'mem': OperandType.memory,
             'phrase': OperandType.phrase,
             'displ': OperandType.displacement,
             'imm': OperandType.immediate,
             'far': OperandType.far,
             'near': OperandType.near,
             'idpspec0': OperandType.specific0,
             'idpspec1': OperandType.specific1,
             'idpspec2': OperandType.specific2,
             'idpspec3': OperandType.specific3,
             'idpspec4': OperandType.specific4,
             'idpspec5': OperandType.specific5}


class OperandBackendQBinDiff(object):
    def __init__(self, data):
        self._data = data

    @property
    def type(self):
        return map_table[self._data['type']]

    @property
    def expressions(self):
        yield from self._data["expr"]

    def __str__(self):
        return ''.join(y['value'] for y in self._data['expr'])


class InstructionBackendQBinDiff(object):
    def __init__(self, data):
        self._data = data

    @property
    def addr(self):
        return self._data['addr']

    @property
    def mnemonic(self):
        return str(self._data['mnem'])

    @property
    def operands(self):
        return [Operand(LoaderType.qbindiff, x) for x in self._data['opnds']]

    @property
    def groups(self):
        return self._data['groups']

    @property
    def comment(self):
        return ""  # TODO: Adding it to the export

    def __str__(self):
        return "%s %s" % (self.mnemonic, ', '.join((str(op) for op in self.operands)))


class FunctionBackendQBinDiff(object):
    def __init__(self, fun, data):
        self._function = fun
        self.addr = None
        self.graph = networkx.DiGraph()
        self.from_json(data)
        self.parents = set()
        self.children = set()

    def from_json(self, data):
        self.addr = data["addr"]
        for node in data["nodes"]:
            self.graph.add_node(node['id'])
            bb = []
            for inst in node['instructions']:
                bb.append(Instruction(LoaderType.qbindiff, inst))
            self._function[node['id']] = bb
        for edge in data["links"]:
            self.graph.add_edge(edge['source'], edge['target'])

    @property
    def type(self):
        return NotImplemented

    @type.setter
    def type(self, value):
        raise NotImplementedError("function type not implemented for qbindiff backend")

    def is_import(self):
        raise NotImplementedError("is_import not implemented for qbindiff backend")


class ProgramBackendQBinDiff(object):
    def __init__(self, program, directory=None, call_graph=None):
        self._name = None
        self._program = program
        if directory is not None:
            self._from_directory(directory)
        if call_graph is not None:
            self._load_call_graph(call_graph)

    @property
    def name(self):
        return self._name

    def _from_directory(self, directory):
        p = Path(directory)
        self._name = p.name[:-5]
        for file in p.iterdir():
            if os.path.splitext(str(file))[1] != ".json":
                logging.warning("skip file %s (not json)" % file)
            else:
                with open(str(file), "r") as fun_handle:
                    f = Function(LoaderType.qbindiff, json.load(fun_handle))
                    self._program[f.addr] = f

    def _load_call_graph(self, file):
        call_graph = json.load(open(str(file), "r"))
        for node in call_graph["nodes"]:
            node_addr = node['id']
            if node_addr not in self._program:
                print("Error missing node: %x" % node_addr)
        for links in call_graph["links"]:
            src = links['source']
            dst = links['target']
            self._program[src].children.add(dst)
            self._program[dst].parents.add(src)

    def __repr__(self):
        return '<Program:%s>' % self.name
