import json
import os.path
import logging
import networkx

from ml_analysis.loader.instruction import Instruction

class Function(dict):
    def __init__(self, data):
        super(dict, self).__init__()
        self.addr = None
        self.graph = networkx.DiGraph()
        self.from_json(data)
        self.parents = set()
        self.children = set()

    def from_json(self, data):
        self.addr = data["addr"]
        for node in data["nodes"]:
            self.graph.add_node(node['id'])
            bb = []
            for inst in node['instructions']:
                bb.append(Instruction(inst))
            self[node['id']] = bb
        for edge in data["links"]:
            self.graph.add_edge(edge['source'], edge['target'])

    def load_obfuscation_data(self, data):
        total = len(data['inst_status'])
        dead = sum(1 for x in data['inst_status'].values() if x == 1)
        self.obfu_percentage = (dead * 100) / total

    def __repr__(self):
        return '<Function: 0x%x>' % self.addr
