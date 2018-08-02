
from qbindiff.features.visitor import OperandFeatureExtractor

class LibName(OperandFeatureExtractor):
    ''' Call to library (local function) '''
    name = "libname"
    key = "lib"

    def call(self, env, expr, full_op=None):
        if expr['type'] == 'libname':
            env.inc_feature(expr['value'])
    
class DatName(OperandFeatureExtractor):
    ''' Reference to specific data in the program '''
    name = "datname"
    key = 'dat'

    def call(self, env, expr, full_op=None):
        if expr['type'] == 'datname':
            env.inc_feature(expr['value'])


class Constant(OperandFeatureExtractor):
    ''' Reference to specific data in the program '''
    name = "cstname"
    key = 'cst'

    def call(self, env, expr, full_op=None):
        if expr['type'] == "number":
            try:
                raw_val = expr['value']
                if raw_val[-1] == "h":
                    val = int(raw_val[:-1], 16)
                else:
                    val = int(raw_val)
                if 0xFFFF < val <= 0xFFFFFF00:
                    if val not in [0x80000000]:
                        env.inc_feature('cst_0x%x' % val)
            except ValueError:
                print('Invalid constant: %s' % (expr['value']))


class ImpName(OperandFeatureExtractor):
    ''' Reference import to an external library '''
    name = 'impname'
    key = 'imp'

    def call(self, env, expr, full_op=None):
        if expr['type'] == 'impname':
            env.inc_feature(expr['value'])
