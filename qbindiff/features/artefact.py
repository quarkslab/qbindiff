
from qbindiff.features.visitor import OperandFeature, Environment, InstructionFeature, ExpressionFeature
from qbindiff.loader.operand import Operand, Expr
from qbindiff.loader.instruction import Instruction


class Address(InstructionFeature):
    """ Address of the function as a feature"""
    name = 'address'
    key = 'addr'

    def visit_instruction(self, instruction: Instruction, env: Environment) -> None:
        value = instruction.addr
        env.add_feature(self.key, value)


class AddressIndex(ProgramFeature, FunctionFeature):
    """ Address index of the function as a feature"""
    key = "address_index"
    key = "addridx"

    def visit_program(self, program: Program, env: Environment):
        self._function_idx = 0

    def visit_function(self, function: Function, env: Environment):
        env.add_feature(self.key, self._function_idx)
        self._function_idx += 1


class LibName(ExpressionFeature):
    """Call to library functions (local function)"""
    name = "libname"
    key = "lib"

    def visit_expression(self, expr: Expr, env: Environment):
        if expr['type'] == 'libname':
            env.inc_feature(expr['value'])


class DatName(ExpressionFeature):
    """References to data in the instruction"""
    name = "datname"
    key = 'dat'

    def visit_expression(self, expr: Expr, env: Environment) -> None:
        if expr['type'] == 'datname':
            env.inc_feature(expr['value'])


class ImpName(ExpressionFeature):
    """References to imports in the instruction"""
    name = 'impname'
    key = 'imp'

    def visit_expression(self, expr: Expr, env: Environment) -> None:
        if expr['type'] == 'impname':
            env.inc_feature(expr['value'])


class Constant(ExpressionFeature):
    """Constant (32/64bits) in the instruction (not addresses)"""
    name = "cstname"
    key = 'cst'

    def visit_expression(self, expr: Expr, env: Environment) -> None:
        if expr['type'] == "number":
            try:
                val = expr['value']
                if isinstance(val, str):
                    val = int(val[:-1], 16) if val[-1] == "h" else int(val)
                if 0xFFFF < val <= 0xFFFFFF00:
                    if val not in [0x80000000]:
                        env.inc_feature('cst_0x%x' % val)
            except ValueError:
                print('Invalid constant: %s' % (expr['value']))

