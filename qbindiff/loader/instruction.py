
class Instruction(object):
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
        return self._data['opnds']

    @property
    def groups(self):
        return self._data['groups']

    @property
    def full_line(self):
        try:
            return self._data['full_line'].encode("utf-8")
        except UnicodeDecodeError:
            print("Cannot decode 0x%x" % self.addr)
            return ""
        return self._data

    def __str__(self):
        operands = [''.join(y['value'] for y in op['expr']) for op in self.operands]
        return "%s %s" % (self.mnemonic, ', '.join(operands))

    def __repr__(self):
        return "<Inst:%s>" % str(self)
