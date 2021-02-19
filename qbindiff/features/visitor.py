from typing import List, Any, Iterable, Dict, Union, Callable

from qbindiff.loader import Program, Function, BasicBlock, Instruction, Operand, Expr


class Environment(object):
    """
    Dict wrapper, representing features where the feature name is the key
    and the value is an integer summing the number of occurrence of that
    feature.
    """
    def __init__(self):
        self.features: Dict[str, Union[int, Dict[str, int]]] = {}

    def add_feature(self, key: str, value: Union[int, float]) -> None:
        self.features[key] = value

    def inc_feature(self, key: str) -> None:
        try:
            self.features[key] += 1
        except KeyError:
            self.features[key] = 1

    def add_dict_feature(self, name: str, key: str, value: Union[int, float]) -> None:
        if name not in self.features:
            self.features[name] = {}
        d = self.features[name]
        d[key] = value

    def inc_dict_feature(self, name: str, key: str) -> None:
        if name not in self.features:
            self.features[name] = {}
        d = self.features[name]
        try:
            d[key] += 1
        except KeyError:
            d[key] = 1


class Visitor(object):
    """
    Abstract class representing interface that a visitor
    must implements to work with a Differ object.
    """

    def visit_item(self, item: Any, env: Environment) -> None:
        """
        Abstract method meant to perform the visit of the item.
        It receives an environment in parameter that is meant to be filled.

        :param item: item to be visited
        :param env: Environment to fill during the visit
        """
        raise NotImplementedError()

    def visit(self, it: Iterable[Any]) -> List[Environment]:
        """
        Function performing the iteration of all items to visit.
        For each of them call visit_item with an environment meant
        to be filled.

        :param it: iterator of items.
        :return: List of environments for all items
        """
        envs = []
        for item in it:
            env = Environment()
            self.visit_item(item, env)
            envs.append(env)
        return envs


class Feature(object):
    """
    Abstract class that represent a feature extractor which sole contraints
    are to define name, key and a function call that is to be called by the
    visitor.
    """
    name = ""
    key = ""

    def __init__(self, weight: float = 1.):
        self._weight = weight

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, value: float) -> None:
        self._weight = value


class ProgramFeature(Feature):
    def visit_program(self, program: Program, env: Environment) -> None:
        pass


class FunctionFeature(Feature):
    def visit_function(self, function: Function, env: Environment) -> None:
        pass


class BasicBlockFeature(Feature):
    def visit_basic_block(self, basicblock: BasicBlock, env: Environment) -> None:
        pass


class InstructionFeature(Feature):
    def visit_instruction(self, instruction: Instruction, env: Environment) -> None:
        pass


class OperandFeature(Feature):
    def visit_operand(self, operand: Operand, env: Environment) -> None:
        pass


class ExpressionFeature(Feature):
    def visit_expression(self, expr: Expr, env: Environment) -> None:
        pass


class ProgramVisitor(Visitor):
    """
    Class aiming at providing a generic program visitor which calls
    the different feature extractor on the appropriate items.
    """
    def __init__(self):
        self.features = {}
        self.program_callbacks = []
        self.function_callbacks = []
        self.basic_block_callbacks = []
        self.instruction_callbacks = []
        self.operand_callbacks = []
        self.expression_callbacks = []

    def visit_item(self, item: Any, env: Environment) -> None:
        """
        Visit a program item according to its type.

        :param item: Can be a Program, Function, Instruction etc..
        :param env: Environment to be filled
        """
        if isinstance(item, Program):
            self.visit_program(item, env)
        elif isinstance(item, Function):
            self.visit_function(item, env)
        elif isinstance(item, BasicBlock):
            self.visit_basic_block(item, env)
        elif isinstance(item, Instruction):
            self.visit_instruction(item, env)
        elif isinstance(item, Operand):
            self.visit_operand(item, env)
        elif isinstance(item, Expr):
            self.visit_expression(item, env)

    def register_feature(self, ft: Feature, weight: float = 1.) -> None:
        """
        Register an instanciated feature extractor on the visitor.
        :param ft: Feature extractor instance
        :param weight: Weight to apply to the feature
        :return: None
        """
        assert(isinstance(ft, Feature))
        if isinstance(ft, ProgramFeature):
            self.register_program_feature_callback(ft.visit_program)
        if isinstance(ft, FunctionFeature):
            self.register_function_feature_callback(ft.visit_function)
        if isinstance(ft, BasicBlockFeature):
            self.register_basic_block_feature_callback(ft.visit_basic_block)
        if isinstance(ft, InstructionFeature):
            self.register_instruction_feature_callback(ft.visit_instruction)
        if isinstance(ft, OperandFeature):
            self.register_operand_feature_callback(ft.visit_operand)
        if isinstance(ft, ExpressionFeature):
            self.register_expression_feature_callback(ft.visit_expression)
        self.features[ft.key] = ft

    def register_program_feature_callback(self, callback: Callable) -> None:
        self.program_callbacks.append(callback)

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

    def visit_program(self, program: Program, env: Environment) -> None:
        """
        Visit the given program with the feature extractor registered beforehand
        with register_feature.
        :param program: program to visit
        :param env: Environment to fill
        :return: ProgramFeatures (dict: Addr-> FunctionFeatures)
        """
        # Call all features attached to the program
        for cb in self.program_callbacks:
            cb(program, env)

        # Recursively call visit on all functions
        for fun in program:
            self.visit_function(fun, env)

    def visit_function(self, func: Function, env: Environment) -> None:
        """
        Visit the given function with the feature extractors registered beforehand.
        :param func: Function to visit
        :param env: Environment to fill
        :return: FunctionFeatures, the features of the function
        """
        # Call all callbacks attacked to a function
        for callback in self.function_callbacks:
            callback(func, env)

        # Recursively call visit for all basic blocks
        for bb in func:
            self.visit_basic_block(bb, env)

    def visit_basic_block(self, basic_block: BasicBlock, env: Environment) -> None:
        """
        Visit the given basic block with the feature extractors registered beforehand.
        :param basic_block: Basic Block to visit
        :param env: Environment to fill
        """
        # Call all callbacks attacked to a basic block
        for callback in self.basic_block_callbacks:
            callback(basic_block, env)

        # Recursively call visit for all instructions
        for inst in basic_block:
            self.visit_instruction(inst, env)

    def visit_instruction(self, instruction: Instruction, env: Environment) -> None:
        """
        Visit the instruction with the feature extractor registered beforehand. The visit
        does not yield new features but update the given environment
        :param instruction: Instruction to visit
        :param env: Environment
        :return: None (perform side effects on the Environment
        """
        # Call all callbacks attached to an instruction
        for callback in self.instruction_callbacks:
            callback(instruction, env)
        # Recursively iter on all operands if there are any features registered
        if self.operand_callbacks:
            for op in instruction.operands:
                self.visit_operand(op, env)

    def visit_operand(self, operand: Operand, env: Environment) -> None:
        """
        Visit the given operand and update the environment accordingly.
        :param operand: Operand
        :param env: Environment
        :return: None
        """
        # Call all callbacks attached to an operand
        for callback in self.operand_callbacks:
            callback(operand, env)
        if self.expression_callbacks:
            for exp in operand.expressions:
                self.visit_expression(exp, env)

    def visit_expression(self, expression: Expr, env: Environment) -> None:
        """
        Visit the given operand and update the environment accordingly.
        :param expression: Expression object to visit
        :param env: Environment
        """
        # Call all callbacks attached to an expression
        for callback in self.expression_callbacks:
            callback(expression, env)

    def get_feature(self, key: str) -> Feature:
        return self.features[key]
