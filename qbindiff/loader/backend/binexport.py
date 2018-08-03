import logging
import networkx
from pathlib import Path

from qbindiff.loader.backend.binexport2_pb2 import BinExport2
from qbindiff.loader.types import OperandType, FunctionType
from qbindiff.loader.function import Function
from qbindiff.loader.instruction import Instruction
from qbindiff.loader.operand import Operand
from qbindiff.loader.types import LoaderType

'''
This is all the types for ida_lines things that can be put on a line !
default  #
regcmt   # comment (not repeatable)
rptcmt   # repeatable comment
autocmt  # auto comment (also used for demangled function comments)
insn     # instruction mnemonic
datname  # name of data
dname    # name of data (function pointer for exception ??) mangled name ?
demname  # demangled name ?
symbol   # symbol "operator": '$+', '(', ')[', '*2+', '*2-', '*2]', '*4+', '*4-', '*4]', '*8+', '*8-', '*8]', '+', '-', ':', ':(', '[', ']'
char     #
string   #
number   # number 
voidop   # no meant to occur ?
cref     #
dref     #
creftail #
dreftail #
error    # on some integer in instruction ? (don't know why..)
prefix   # instruction prefixes (lock, rep, repne)
binpref  #
extra    #
altop    #
hidname  # code name location (seen for SEH handler)
libname  # library function (local) eg: _strcpy etc...
locname  # named variable (shadow an offset or reg+off) eg: var_XX, arg_XX or struct fields
codname  # location name eg: loc_XXX, or sub_XXX
asmdir   #
macro    #
dstr     #
dchar    #
dnum     # dummy number ? (used in nop XXXXX 0000h (to align))
keyword  # keywords eg: 'byte ptr', 'dword ptr', 'large ', 'offset', 'qword ptr', 'word ptr', 'xmmword ptr'
reg      # register (include ss: ds: etc.)
impname  # imported function
segname  #
unkname  # data (with unknown type) eg: unk_XXX
cname    # loc name (for known functions)
uname    #
collapsed#
addr     #
unk      # 3rd operand on x86 (created by me for not recognized operands)
'''

'''
SYMBOL = 1
    "__alloca_probe"  # call to functions
    "??_7bad_alloc@std@@6B@" # symbol in rdata

     --> tester si la string est dans les fun_name si oui le type etc..
IMMEDIATE_INT = 2
IMMEDIATE_FLOAT = 3
OPERATOR = 4
    "+"
    'ds:'
    'ss:'
REGISTER         # register (in symbol)
SIZE_PREFIX = 6  # b1, b2, b4 or b8
DEREFERENCE = 7
    - Immediate: (voir si reference dans data_refs)
    -

DataReference:


Reference address_comment:
    -> s'il n'est pas dans les symbols des expr ni nom de fonction
    alors commentaire
'''


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
        yield from (self._program.proto.expression[idx] for idx in self._me().expression_index)

    @property
    def expressions(self):
        is_deref = False
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
                is_deref = True
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
    def type(self):
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

    def __str__(self):
        return ''.join(self._program.proto.expression[idx].symbol for idx in self._me().expression_index)

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
        return [Operand(LoaderType.binexport, self._program, self._function, self, op_idx) for op_idx in self._me().operand_index]

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
    def __init__(self, function, program, data_refs, addr_refs, pb_fun, is_import=False, addr=None):
        self._function = function
        self.addr = addr
        self.parents = set()
        self.children = set()
        self.graph = networkx.DiGraph()
        self._pb_type = None  # Set by the Program constructor
        self.name = None  # Set by the Program constructor (mangled name)

        if is_import:
            return

        self.addr = self._get_basic_block_addr(program, pb_fun.entry_basic_block_index)

        cur_addr = None

        # Load the different basic blocks
        for bb_idx in pb_fun.basic_block_index:
            bb = program.proto.basic_block[bb_idx]
            bb_addr = None
            bb_data = []
            for rng in bb.instruction_index:
                for idx in range(rng.begin_index, rng.end_index if rng.end_index else rng.begin_index+1):
                    pb_i = program.proto.instruction[idx]

                    # addresses computation
                    if cur_addr is None:  # once per function in theory
                        if pb_i.address == 0:
                            tmp_sz = 0
                            tmp_idx = idx
                            while True:
                                tmp_idx -= 1
                                tmp_sz += len(program.proto.instruction[tmp_idx].raw_bytes)
                                if program.proto.instruction[tmp_idx].address != 0:
                                    break
                            cur_addr = program.proto.instruction[tmp_idx].address + tmp_sz
                            logging.debug("current address unset: backtracked 0x%x up to: 0x%x" % (cur_addr, cur_addr-tmp_sz))
                    if pb_i.address != 0:
                        cur_addr = pb_i.address
                    if bb_addr is None:
                        bb_addr = cur_addr

                    inst = Instruction(LoaderType.binexport, program, self, cur_addr, idx)

                    bb_data.append(inst)
                    if idx in data_refs:  # Add some
                        inst._backend.data_refs = data_refs[idx]
                    if idx in addr_refs:
                        inst._backend.addr_refs = addr_refs[idx]

                    cur_addr += len(pb_i.raw_bytes)
            self._function[bb_addr] = bb_data
            self.graph.add_node(bb_addr)

        if len(pb_fun.basic_block_index) != len(self._function):
            print("%x, bb:%d, self:%d" % (self.addr, len(pb_fun.basic_block_index), len(self._function)))

        # Load the edges between blocks
        for edge in pb_fun.edge:
            bb_src = self._get_basic_block_addr(program, edge.source_basic_block_index)
            bb_dst = self._get_basic_block_addr(program, edge.target_basic_block_index)
            self.graph.add_edge(bb_src, bb_dst)

    def _get_basic_block_addr(self, program, idx):
        return program.proto.instruction[program.proto.basic_block[idx].instruction_index[0].begin_index].address

    @property
    def function(self):
        return self._function

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
        self._pb.ParseFromString(Path(file).read_bytes())
        self._mask = 0xFFFFFFFF if self.architecture.endswith("32") else 0xFFFFFFFFFFFFFFFF
        self.fun_names = {}

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
            if node.address not in self._program and node.type == cg.Vertex.IMPORTED:
                self._program[node.address] = Function(LoaderType.binexport, self, data_refs, addr_refs, None, is_import=True, addr=node.address)
                count_imp += 1
            if node.address not in self._program and node.type == cg.Vertex.NORMAL:
                logging.error("Missing function address: 0x%x (%d)" % (node.address, node.type))

            self._program[node.address].type = node.type
            self._program[node.address].name = node.mangled_name
        for edge in cg.edge:
            src = cg.vertex[edge.source_vertex_index].address
            dst = cg.vertex[edge.target_vertex_index].address
            self._program[src].children.add(dst)
            self._program[dst].parents.add(src)

        for f in self._program.values():  # Create a map of function names for quick lookup later on
            self.fun_names[f.name] = f

        logging.debug("total all:%d, imported:%d collision:%d (total:%d)" % (count_f, count_imp, coll, (count_f+count_imp+coll)))

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
