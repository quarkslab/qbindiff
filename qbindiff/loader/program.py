import json
import os.path
from pathlib import Path
import logging
from collections import OrderedDict
logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] %(message)s")


from qbindiff.loader.function import Function


class Program(OrderedDict):
    def __init__(self, directory=None, loader="qbindiff"):
        super(dict, self).__init__()
        self.name = None
        #TODO: implement a loader for binexport !
        if directory is not None:
            self.from_directory(directory)

    def from_directory(self, directory):
        p = Path(directory)
        self.name = p.name[:-5]
        for file in p.iterdir():
            if os.path.splitext(str(file))[1] != ".json":
                logging.warning("skip file %s (not json)" % file)
            else:
                with open(str(file), "r") as fun_handle:
                    #logging.debug("Load fun: %s" % file.name)
                    f = Function(json.load(fun_handle))
                    self[f.addr] = f

    def load_call_graph(self, file):
        call_graph = json.load(open(str(file), "r"))
        for node in call_graph["nodes"]:
            node_addr = node['id']
            if node_addr not in self:
                print("Error missing node: %x" % node_addr)
        for links in call_graph["links"]:
            src = links['source']
            dst = links['target']
            self[src].children.add(dst)
            self[dst].parents.add(src)

    def load_obfuscation_data(self, file):
        all_data = json.load(open(file, "r"))
        for fun_addr, data in all_data.items():
            self[int(fun_addr)].load_obfuscation_data(data)

    def __repr__(self):
        return '<Program:%s>' % self.name
