
from qbindiff.features.visitor import OperandFeatureExtractor


class LibName(OperandFeatureExtractor):
    """Call to library functions (local function)"""
    name = "libname"
    key = "lib"

    def call(self, env, expr, full_op=None):
        if expr['type'] == 'libname':
            env.inc_feature(expr['value'])


class DatName(OperandFeatureExtractor):
    """References to data in the instruction"""
    name = "datname"
    key = 'dat'

    def call(self, env, expr, full_op=None):
        if expr['type'] == 'datname':
            env.inc_feature(expr['value'])


class Constant(OperandFeatureExtractor):
    """Constant (32/64bits) in the instruction (not addresses)"""
    name = "cstname"
    key = 'cst'

    def call(self, env, expr, full_op=None):
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


class ImpName(OperandFeatureExtractor):
    """References to imports in the instruction"""
    name = 'impname'
    key = 'imp'

    def call(self, env, expr, full_op=None):
        if expr['type'] == 'impname':
            env.inc_feature(expr['value'])
