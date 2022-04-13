from typing import List, Any, Iterable, Dict, Union, Callable
from collections import defaultdict

from qbindiff.loader import Program, Function, BasicBlock, Instruction, Operand, Expr


class FeatureCollector:
    """
    Dict wrapper, representing a collection of features where the key is the feature
    name and the value is the feature score which can be either a number or a dict.
    """

    def __init__(self):
        self._features: Dict[str, Union[int, float, Dict[str, Union[int, float]]]] = {}

    def add_feature(self, key: str, value: Union[int, float]) -> None:
        self._features.setdefault(key, 0)
        self._features[key] += value

    def add_dict_feature(self, key: str, value: Dict[str, Union[int, float]]) -> None:
        self._features.setdefault(key, defaultdict(int))
        for k, v in value.items():
            self._features[key][k] += v

    def to_vector(self) -> List:
        """Transform the collection to a feature vector"""
        raise NotImplementedError()


class Visitor:
    """
    Abstract class representing interface that a visitor
    must implements to work with a Differ object.
    """

    def visit(self, prog: Program) -> Dict[int, FeatureCollector]:
        """
        Function performing the visit on the Program by calling visit_item with a
        FeatureCollector meant to be filled.

        :param prog: the Program.
        :return: A Dict in which keys are function addresses and values are the FeatureCollector
        """
        functionsFeatures = {}
        for func in prog:
            collector = FeatureCollector()
            self.visit_item(func, collector)
            functionsFeatures[func.addr] = collector
        return functionsFeatures

    def visit_item(self, item: Any, collector: FeatureCollector) -> None:
        """
        Abstract method meant to perform the visit of the item.
        It receives an environment in parameter that is meant to be filled.

        :param item: item to be visited
        :param collector: FeatureCollector to fill during the visit
        """
        raise NotImplementedError()

    def feature_keys(self) -> List[str]:
        raise NotImplementedError()

    def feature_weight(self, key: str) -> float:
        raise NotImplementedError()


class FeatureExtractor:
    """
    Abstract class that represent a feature extractor which sole contraints are to
    define name, key and a function call that is to be called by the visitor.
    """

    name = ""
    key = ""

    def __init__(self, weight: float = 1.0):
        self._weight = weight

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, value: float) -> None:
        self._weight = value


class FunctionFeatureExtractor(FeatureExtractor):
    def visit_function(self, function: Function, collector: FeatureCollector) -> None:
        raise NotImplementedError()


class BasicBlockFeatureExtractor(FeatureExtractor):
    def visit_basic_block(
        self, basicblock: BasicBlock, collector: FeatureCollector
    ) -> None:
        raise NotImplementedError()


class InstructionFeatureExtractor(FeatureExtractor):
    def visit_instruction(
        self, instruction: Instruction, collector: FeatureCollector
    ) -> None:
        raise NotImplementedError()


class OperandFeatureExtractor(FeatureExtractor):
    def visit_operand(self, operand: Operand, collector: FeatureCollector) -> None:
        raise NotImplementedError()


class ExpressionFeatureExtractor(FeatureExtractor):
    def visit_expression(self, expr: Expr, collector: FeatureCollector) -> None:
        raise NotImplementedError()


class ProgramVisitor(Visitor):
    """
    Class aiming at providing a generic program visitor which calls
    the different feature extractor on the appropriate items.
    """

    def __init__(self):
        self.feature_extractors = {}
        self.function_callbacks = []
        self.basic_block_callbacks = []
        self.instruction_callbacks = []
        self.operand_callbacks = []
        self.expression_callbacks = []

    def visit_item(self, item: Any, collector: FeatureCollector) -> None:
        """
        Visit a program item according to its type.

        :param item: Can be a Function, Instruction etc..
        :param collector: FeatureCollector to be filled
        """
        if isinstance(item, Function):
            self.visit_function(item, collector)
        elif isinstance(item, BasicBlock):
            self.visit_basic_block(item, collector)
        elif isinstance(item, Instruction):
            self.visit_instruction(item, collector)
        elif isinstance(item, Operand):
            self.visit_operand(item, collector)
        # elif isinstance(item, Expr):
        elif isinstance(item, dict):
            self.visit_expression(item, collector)

    def register_feature_extractor(self, fte: FeatureExtractor) -> None:
        """
        Register an instanciated feature extractor on the visitor.

        :param ft: Feature extractor instance
        :param weight: Weight to apply to the feature
        :return: None
        """
        assert isinstance(fte, FeatureExtractor)
        if isinstance(fte, FunctionFeatureExtractor):
            self.register_function_feature_callback(fte.visit_function)
        if isinstance(fte, BasicBlockFeatureExtractor):
            self.register_basic_block_feature_callback(fte.visit_basic_block)
        if isinstance(fte, InstructionFeatureExtractor):
            self.register_instruction_feature_callback(fte.visit_instruction)
        if isinstance(fte, OperandFeatureExtractor):
            self.register_operand_feature_callback(fte.visit_operand)
        if isinstance(fte, ExpressionFeatureExtractor):
            self.register_expression_feature_callback(fte.visit_expression)
        self.feature_extractors[fte.key] = fte

    def register_function_feature_callback(self, callback: Callable) -> None:
        self.function_callbacks.append(callback)

    def register_basic_block_feature_callback(self, callback: Callable) -> None:
        self.basic_block_callbacks.append(callback)

    def register_instruction_feature_callback(self, callback: Callable) -> None:
        self.instruction_callbacks.append(callback)

    def register_operand_feature_callback(self, callback: Callable) -> None:
        self.operand_callbacks.append(callback)

    def register_expression_feature_callback(self, callback: Callable) -> None:
        self.expression_callbacks.append(callback)

    def visit_function(self, func: Function, collector: FeatureCollector) -> None:
        """
        Visit the given function with the feature extractors registered beforehand.

        :param func: Function to visit
        :param collector: FeatureCollector to fill
        """
        # Call all callbacks attacked to a function
        for callback in self.function_callbacks:
            if not func.is_import():
                callback(func, collector)

        # Recursively call visit for all basic blocks
        for bb in func:
            self.visit_basic_block(bb, collector)

    def visit_basic_block(
        self, basic_block: BasicBlock, collector: FeatureCollector
    ) -> None:
        """
        Visit the given basic block with the feature extractors registered beforehand.

        :param basic_block: Basic Block to visit
        :param collector: FeatureCollector to fill
        """
        # Call all callbacks attacked to a basic block
        for callback in self.basic_block_callbacks:
            callback(basic_block, collector)

        # Recursively call visit for all instructions
        for inst in basic_block:
            self.visit_instruction(inst, collector)

    def visit_instruction(
        self, instruction: Instruction, collector: FeatureCollector
    ) -> None:
        """
        Visit the instruction with the feature extractor registered beforehand.

        :param instruction: Instruction to visit
        :param collector: FeatureCollector to fill
        """
        # Call all callbacks attached to an instruction
        for callback in self.instruction_callbacks:
            callback(instruction, collector)

        # Recursively call visit for all operands
        for op in instruction.operands:
            self.visit_operand(op, collector)

    def visit_operand(self, operand: Operand, collector: FeatureCollector) -> None:
        """
        Visit the given operand with the feature extractor registered beforehand.

        :param operand: Operand
        :param collector: FeatureCollector to fill
        """
        # Call all callbacks attached to an operand
        for callback in self.operand_callbacks:
            callback(operand, collector)

        for exp in operand.expressions:
            self.visit_expression(exp, collector)

    def visit_expression(self, expression: Expr, collector: FeatureCollector) -> None:
        """
        Visit the given operand with the feature extractor registered beforehand.

        :param expression: Expression object to visit
        :param collector: FeatureCollector
        """
        # Call all callbacks attached to an expression
        for callback in self.expression_callbacks:
            callback(expression, collector)

    def feature_keys(self) -> List[str]:
        return list(self.features.keys())

    def feature_weight(self, key: str) -> float:
        return self.features[key].weight
