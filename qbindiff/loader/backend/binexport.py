from __future__ import absolute_import
import logging
import networkx

from qbindiff.loader.backend.binexport2_pb2 import BinExport2
from qbindiff.loader.types import OperandType, FunctionType
from qbindiff.loader.function import Function
from qbindiff.loader.instruction import Instruction
from qbindiff.loader.operand import Operand
from qbindiff.loader.types import LoaderType


# === General purpose binexport functions ===
def _get_instruction_address(pb, inst_idx):
    inst = pb.instruction[inst_idx]
    if inst.HasField('address'):
        return inst.address
    else:
        return _backtrack_instruction_address(pb, inst_idx)


def _backtrack_instruction_address(pb, idx):
    tmp_sz = 0
    tmp_idx = idx
    if tmp_idx == 0:
        return pb.instruction[tmp_idx].address
    while True:
        tmp_idx -= 1
        tmp_sz += len(pb.instruction[tmp_idx].raw_bytes)
        if pb.instruction[tmp_idx].HasField('address'):
            break
    return pb.instruction[tmp_idx].address + tmp_sz


def _get_basic_block_addr(pb, bb_idx):
    inst = pb.basic_block[bb_idx].instruction_index[0].begin_index
    return _get_instruction_address(pb, inst)


# ===========================================


class OperandBackendBinexport:

    __sz_lookup = {'b1': 1, 'b2': 2, 'b4': 4, 'b8': 8, 'b10': 10, 'b16': 16, 'b32': 32, 'b64': 64}
    __sz_name = {1: 'byte', 2: 'word', 4: 'dword', 8: "qword", 10: 'b10', 16: "xmmword", 32: "ymmword", 64: "zmmword"}

    def __init__(self, program, fun, inst, op_idx):
        self._program = program
        self._function = fun
        self._instruction = inst
        self._idx = op_idx

    def _me(self):
        return self._program.proto.operand[self._idx]

    def _pb_expressions(self):
        for it in (self._program.proto.expression[idx] for idx in self._me().expression_index):
            yield it

    @property
    def expressions(self):
        # is_deref = False
        size = None
        for exp in self._pb_expressions():
            if exp.type == BinExport2.Expression.SYMBOL:  # If the expression is a symbol
                if exp.symbol in self._program.fun_names:  # If it is a function name
                    f = self._program.fun_names[exp.symbol]
                    if f.type == FunctionType.normal:
                        yield {'type': 'codname', 'value': exp.symbol}
                    elif f.type == FunctionType.library:
                        yield {'type': 'libname', 'value': exp.symbol}
                    elif f.type == FunctionType.imported:
                        yield {'type': 'impname', 'value': exp.symbol}
                    elif f.type == FunctionType.thunk:
                        yield {'type': 'cname', 'value': exp.symbol}
                    else:
                        pass  # invalid fucntion type just ignore it
                else:
                    yield {'type': 'locname', 'value': exp.symbol}  # for var_, arg_

            elif exp.type == BinExport2.Expression.IMMEDIATE_INT:  # If the expression is an immediate
                if exp.immediate in self._instruction.data_refs:
                    # TODO: (near future) using the addr_refs to return the symbol
                    s = "%s_%X" % (self.__sz_name[size], exp.immediate)
                    yield {'type': 'datname', 'value': s}
                else:
                    if exp.immediate in self._program.program:  # if it is a function
                        yield {'type': 'codname', 'value': "sub_%X" % exp.immediate}
                    elif exp.immediate in self._function.function:  # its a basic block address
                        yield {'type': 'codname', 'value': 'loc_%X' % exp.immediate}
                    else:
                        yield {'type': 'number', 'value': self._program.addr_mask(exp.immediate)}

            elif exp.type == BinExport2.Expression.IMMEDIATE_FLOAT:
                print("IMMEDIATE FLOAT ignored:", exp)
            elif exp.type == BinExport2.Expression.OPERATOR:
                yield {'type': 'symbol', 'value': exp.symbol}
            elif exp.type == BinExport2.Expression.REGISTER:
                yield {'type': 'reg', 'value': exp.symbol}
            elif exp.type == BinExport2.Expression.DEREFERENCE:
                yield {'type': 'symbol', 'value': exp.symbol}
                # is_deref = True
            elif exp.type == BinExport2.Expression.SIZE_PREFIX:
                size = self.__sz_lookup[exp.symbol]
            else:
                print("woot:", exp)

    def byte_size(self):
        exp = self._program.proto.expression[self._me().expression_index[0]]
        if exp.type == BinExport2.Expression.SIZE_PREFIX:
            return self.__sz_lookup[exp.symbol]
        else:
            raise Exception("First expression not byte size..")

    @property
    def type(self) -> OperandType:
        for exp in (self._program.proto.expression[idx] for idx in self._me().expression_index):
            if exp.type == BinExport2.Expression.SIZE_PREFIX:
                continue
            elif exp.type == BinExport2.Expression.SYMBOL:
                return OperandType.memory  # As it is either a ref to data or function
            elif exp.type == BinExport2.Expression.IMMEDIATE_INT:
                return OperandType.immediate  # Could also have been far, near and memory?
            elif exp.type == BinExport2.Expression.IMMEDIATE_FLOAT:
                return OperandType.specific0
            elif exp.type == BinExport2.Expression.OPERATOR:
                continue
            elif exp.type == BinExport2.Expression.REGISTER:
                return OperandType.register
            elif exp.type == BinExport2.Expression.DEREFERENCE:
                return OperandType.displacement  # could also have been phrase
            else:
                print("wooot", exp.type)

        # if we reach here something necessarily went wrong
        if len(self._me().expression_index) == 1 and self._program.architecture.startswith("ARM"):
            if self._program.proto.expression[self._me().expression_index[0]].type == BinExport2.Expression.OPERATOR:
                return OperandType.specific5  # Specific handling of some ARM flags typed as OPERATOR
            else:
                logging.error("Unknown case for operand type on ARM: %s" % str(self))
        else:
            logging.error("No type found for operand: %s" % str(self))

    def __str__(self):
        return ''.join(self._program.proto.expression[idx].symbol for idx in self._me().expression_index if
                       self._program.proto.expression[idx].type != BinExport2.Expression.SIZE_PREFIX)

    def __repr__(self):
        return "<Op:%s>" % str(self)


class InstructionBackendBinExport:
    def __init__(self, program, fun, addr, i_idx):
        self._addr = addr
        self._program = program
        self._function = fun
        self._idx = i_idx
        self.data_refs = []
        self.addr_refs = []

    @property
    def addr(self):
        return self._addr

    @property
    def mnemonic(self):
        return self._program.proto.mnemonic[self._program.proto.instruction[self._idx].mnemonic_index].name

    def _me(self):
        return self._program.proto.instruction[self._idx]

    @property
    def operands(self):
        return [Operand(LoaderType.binexport, self._program, self._function, self, op_idx)
                for op_idx in self._me().operand_index]

    @property
    def groups(self):
        return []  # not supported

    @property
    def comment(self):
        if len(self.data_refs) >= len(self.addr_refs):
            ith = len(self.data_refs)
        else:
            ith = 0
        if self.addr_refs[ith:]:
            last = self.addr_refs[-1]
            if self.is_function_entry():
                if last == self._program[self.addr].name:
                    try:
                        return self.addr_refs[-2]
                    except IndexError:
                        return ""
            else:
                return last
        else:
            return ""

    def is_function_entry(self):
        return self.addr in self._program


class FunctionBackendBinExport(object):
    def __init__(self, fun, program, data_refs, addr_refs, pb_fun, is_import=False, addr=None):
        self._function = fun
        self.addr = addr
        self.parents = set()
        self.children = set()
        self._graph = networkx.DiGraph()
        self._pb_type = None  # Set by the Program constructor
        self._name = None  # Set by the Program constructor (mangled name)

        if is_import:
            return

        self.addr = _get_basic_block_addr(program.proto, pb_fun.entry_basic_block_index)

        cur_addr = None
        prev_idx = -2
        tmp_mapping = {}
        bb_count = 0
        for bb_idx in pb_fun.basic_block_index:
            for rng in program.proto.basic_block[bb_idx].instruction_index:  # Ranges are in fact the true basic blocks!
                bb_count += 1
                bb_addr = None
                bb_data = []
                for idx in range(rng.begin_index, (rng.end_index if rng.end_index else rng.begin_index+1)):

                    if idx != prev_idx+1:  # if the current idx is different from the previous range or bb
                        cur_addr = None  # reset the addr has we have no guarantee on the continuity of the address

                    pb_inst = program.proto.instruction[idx]

                    if pb_inst.HasField('address'):  # If the instruction have an address set (can be 0)
                        if cur_addr is not None and cur_addr != pb_inst.address:
                            # logging.warning("cur_addr different from inst address: %x != %x (%d) (%d->%d)" %
                            #                                    (cur_addr, pb_inst.address, bb_idx, prev_idx, idx))
                            pass    # might be legit if within the basic block there is data
                                    # thus within the same range not contiguous address can co-exists
                        cur_addr = pb_inst.address  # set the address to the one of inst regardless cur_addr was set
                    else:
                        if not cur_addr:  # if cur_addr_not set backtrack to get it
                            cur_addr = _get_instruction_address(program.proto, idx)

                    # At this point we should have a cur_addr correctly set to the right instruction address
                    if not bb_addr:
                        bb_addr = cur_addr

                    # At this point do the instruction initialization
                    inst = Instruction(LoaderType.binexport, program, self, cur_addr, idx)
                    bb_data.append(inst)
                    if idx in data_refs:  # Add some
                        inst._backend.data_refs = data_refs[idx]
                    if idx in addr_refs:
                        inst._backend.addr_refs = addr_refs[idx]

                    cur_addr += len(pb_inst.raw_bytes)  # increment the cur_addr with the address size
                    prev_idx = idx

                if bb_addr in self._function:
                    logging.error("0x%x basic block address (0x%x) already in(idx:%d)" % (self.addr, bb_addr, bb_idx))
                self._function[bb_addr] = bb_data
                tmp_mapping[bb_idx] = bb_addr
                self._graph.add_node(bb_addr)

        if bb_count != len(self._function):
            logging.error("Wrong basic block number %x, bb:%d, self:%d" %
                          (self.addr, len(pb_fun.basic_block_index), len(self._function)))

        # Load the edges between blocks
        for edge in pb_fun.edge:
            bb_src = tmp_mapping[edge.source_basic_block_index]
            bb_dst = tmp_mapping[edge.target_basic_block_index]
            self._graph.add_edge(bb_src, bb_dst)

    @property
    def graph(self) -> networkx.DiGraph:
        return self._graph

    @property
    def function(self):
        return self._function

    @property
    def name(self):
        return self._name if self._name else "sub_%X" % self.addr

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def type(self):
        return {BinExport2.CallGraph.Vertex.NORMAL: FunctionType.normal,
                BinExport2.CallGraph.Vertex.LIBRARY: FunctionType.library,
                BinExport2.CallGraph.Vertex.IMPORTED: FunctionType.imported,
                BinExport2.CallGraph.Vertex.THUNK: FunctionType.thunk,
                BinExport2.CallGraph.Vertex.INVALID: FunctionType.invalid}[self._pb_type]

    @type.setter
    def type(self, value):
        self._pb_type = value

    def is_import(self):
        return self.type == FunctionType.imported


class ProgramBackendBinExport(object):
    def __init__(self, program, file):
        self._program = program
        self._pb = BinExport2()
        with open(file, 'rb') as f:
            self._pb.ParseFromString(f.read())
        self._mask = 0xFFFFFFFF if self.architecture.endswith("32") else 0xFFFFFFFFFFFFFFFF
        self.fun_names = {}
        self._callgraph = networkx.DiGraph()

        # Make the data refs map
        data_refs = {}
        for entry in self.proto.data_reference[::-1]:
            if entry.instruction_index in data_refs:
                data_refs[entry.instruction_index].append(entry.address)
            else:
                data_refs[entry.instruction_index] = [entry.address]

        # Make the address comment
        addr_refs = {}
        for entry in self.proto.address_comment[::-1]:
            if entry.instruction_index in addr_refs:
                addr_refs[entry.instruction_index].append(self.proto.string_table[entry.string_table_index])
            else:
                addr_refs[entry.instruction_index] = [self.proto.string_table[entry.string_table_index]]

        count_f = 0
        coll = 0
        # Load all the functions
        for i, pb_fun in enumerate(self.proto.flow_graph):
            #logging.warning("Parse function idx: %d" % i)
            f = Function(LoaderType.binexport, self, data_refs, addr_refs, pb_fun)
            if f.addr in self._program:
                logging.error("Address collision for 0x%x" % f.addr)
                coll += 1
            self._program[f.addr] = f
            count_f += 1

        count_imp = 0
        # Load the callgraph
        cg = self.proto.call_graph
        for node in cg.vertex:
            self._callgraph.add_node(node.address)
            if node.address not in self._program and node.type == cg.Vertex.IMPORTED:
                self._program[node.address] = Function(LoaderType.binexport, self, data_refs, addr_refs, None,
                                                       is_import=True, addr=node.address)
                count_imp += 1
            if node.address not in self._program and node.type == cg.Vertex.NORMAL:
                logging.error("Missing function address: 0x%x (%d)" % (node.address, node.type))

            self._program[node.address].type = node.type
            self._program[node.address].name = node.mangled_name
        for edge in cg.edge:
            src = cg.vertex[edge.source_vertex_index].address
            dst = cg.vertex[edge.target_vertex_index].address
            self._callgraph.add_edge(src, dst)
            self._program[src].children.add(dst)
            self._program[dst].parents.add(src)

        for f in self._program.values():  # Create a map of function names for quick lookup later on
            self.fun_names[f.name] = f

        logging.debug("total all:%d, imported:%d collision:%d (total:%d)" %
                      (count_f, count_imp, coll, (count_f+count_imp+coll)))

    def addr_mask(self, value):
        return value & self._mask

    @property
    def program(self):
        return self._program

    @property
    def proto(self):
        return self._pb

    @property
    def name(self):
        return self.proto.meta_information.executable_name

    @property
    def architecture(self):
        return self.proto.meta_information.architecture_name

    def __repr_(self):
        return '<Program:%s>' % self.name

    @property
    def callgraph(self) -> networkx.DiGraph:
        return self._callgraph
