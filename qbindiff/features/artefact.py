from qbindiff.features.extractor import (
    FeatureCollector,
    InstructionFeatureExtractor,
    ExpressionFeatureExtractor,
)
from qbindiff.loader import Instruction, Expr


class Address(InstructionFeatureExtractor):
    """Address of the function as a feature"""

    name = "address"
    key = "addr"

    def visit_instruction(
        self, instruction: Instruction, collector: FeatureCollector
    ) -> None:
        value = instruction.addr
        collector.add_feature(self.key, value)


# Currently not implemented
'''class AddressIndex(ProgramFeature, FunctionFeature):
    """Address index of the function as a feature"""

    key = "address_index"
    key = "addridx"

    def visit_program(self, program: Program, collector: FeatureCollector):
        self._function_idx = 0

    def visit_function(self, function: Function, collector: FeatureCollector):
        collector.add_feature(self.key, self._function_idx)
        self._function_idx += 1'''


class LibName(ExpressionFeatureExtractor):
    """Call to library functions (local function)"""

    name = "libname"
    key = "lib"

    def visit_expression(self, expression: Expr, collector: FeatureCollector):
        if expression["type"] == "libname":
            collector.add_dict_feature(self.key, {expression["value"]: 1})


class DatName(ExpressionFeatureExtractor):
    """References to data in the instruction"""

    name = "datname"
    key = "dat"

    def visit_expression(self, expression: Expr, collector: FeatureCollector) -> None:
        if expression["type"] == "datname":
            collector.add_dict_feature(self.key, {expression["value"]: 1})


class ImpName(ExpressionFeatureExtractor):
    """References to imports in the instruction"""

    name = "impname"
    key = "imp"

    def visit_expression(self, expression: Expr, collector: FeatureCollector) -> None:
        if expression["type"] == "impname":
            collector.add_dict_feature(self.key, {expression["value"]: 1})


class Constant(ExpressionFeatureExtractor):
    """Constant (32/64bits) in the instruction (not addresses)"""

    name = "cstname"
    key = "cst"

    def visit_expression(self, expression: Expr, collector: FeatureCollector) -> None:
        if expression["type"] == "number":
            try:
                val = expression["value"]
                if isinstance(val, str):
                    val = int(val[:-1], 16) if val[-1] == "h" else int(val)
                if 0xFFFF < val <= 0xFFFFFF00:
                    if val not in [0x80000000]:
                        collector.add_dict_feature(self.key, {"cst_0x%x" % val: 1})
            except ValueError:
                print("Invalid constant: %s" % (expression["value"]))
