qBinDiff
========

Experimental binary diffing tool based on the **Network Alignemnt Quadratic Problem**.

Installation
------------

qBinDiff follows the regular installation process given below:

    python3 setup.py install

Usage (command line)
--------------------

After installation, the binary ``qbindiff`` is available in the path.
It takes in input two exported files and produce a json file containing
the matching between functions. The default format for input files is
[BinExport](https://github.com/google/binexport). The complete command
line options are:

```bash
Usage: qbindiff [OPTIONS] <primary file> <secondary file>

  qBinDiff is an experimental binary diffing tool based on machine learning technics, namely Belief propagation.

Options:
  -o, --output PATH               Output file matching [default: matching.json]
  -l, --loader <loader>           Input files type between ['qbindiff', 'binexport', 'diaphora']. [default loader: binexport]
  -f, --feature <feature>         The following features are available:
                                  - Gnb, graph_nblock: Number of basic blocks in the function
                                  - Gt, graph_transitivity: Transitivity of the function flow graph
                                  - Gdi,
                                  graph_diameter: Diamater of the function flow graph
                                  - Gnc, graph_num_components: Number of components in the function (non-connected flow graphs)
                                  - Gp, groups_category: Group of
                                  the instruction (FPU, SSE, stack..)
                                  - imp, impname: References to imports in the instruction
                                  - Gd, graph_density: Density of the function flow graph
                                  - Mt, mnemonic_typed:
                                  Mnemonic and type of operand feature
                                  - cst, cstname: Constant (32/64bits) in the instruction (not addresses)
                                  - Gmd, graph_mean_degree: Mean degree of the function
                                  - M, mnemonic:
                                  Mnemonic of instructions feature
                                  - dat, datname: References to data in the instruction
                                  - Gmib, graph_mean_inst_block: Mean of instruction per basic blocks in the function
                                  -
                                  Gcom, graph_community: Number of graph communities (Louvain modularity)
                                  - lib, libname: Call to library functions (local function)
  -d, --distance <function>       Mathematical distance function between cosine and correlation [default: correlation]
  -t, --threshold FLOAT           Distance treshold to keep matches between 0.0 to 1.0 [default: 0.50]
  -i, --maxiter INTEGER           Maximum number of iteration for belief propagation [default: 80]
  --msim INTEGER                  Maximize similarity matching (alpha for NAQP) [default: 1]
  --mcall INTEGER                 Maximize call graph matching (beta for NAQP) [default: 2]
  --refine-match / --no-refine-match
  -v, --verbose                   Activate debugging messages
  -h, --help                      Show this message and exit.
```

Usage library
-------------

The strength of qBinDiff is to be usable as a python library. The following snippet shows an example
of loading to binexport files and to compare them using the mnemonic feature.

```python
from qbindiff.loader.program import Program
from qbindiff.features.mnemonic import MnemonicSimple
from qbindiff.differ.qbindiff import QBinDiff
p1 = Program()
p1.load_binexport("primary.BinExport")
p2 = Program()
p2.load_binexport("secondary.BinExport")
differ = QBinDiff()
differ.register_feature(MnemonicSimple())
differ.threshold = 0.5
differ.maxiter = 50
differ.run(match_refine=True)
matching = differ.matching
```