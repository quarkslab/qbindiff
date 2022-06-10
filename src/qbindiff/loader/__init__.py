from qbindiff.loader.data import Data
from qbindiff.loader.structure import Structure, StructureMember
from qbindiff.loader.operand import Operand
from qbindiff.loader.instruction import Instruction
from qbindiff.loader.function import Function
from qbindiff.loader.basic_block import BasicBlock
from qbindiff.loader.program import Program
from qbindiff.loader.types import LoaderType

LOADERS = {
    "binexport": LoaderType.binexport,
    "qbinexport": LoaderType.qbinexport,
}
