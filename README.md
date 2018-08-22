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

    Usage: qbindiff [OPTIONS] <primary file> <secondary file>
    
      qBinDiff is an experimental binary diffing tool based on machine learning technics, namely Belief
      propagation.
    
    Options:
      -o, --output PATH               Output file matching [default: matching.json]
      -l, --loader <loader>           Input files type between ['qbindiff', 'binexport', 'diaphora', 'ida'].
                                      [default loader: binexport]
      -f, --feature <feature>         The following features are available:
                                      - Gcom, graph_community: Number
                                      of graph communities (Louvain modularity)
                                      - lib, libname: Call to
                                      library functions (local function)
                                      - Gnb, graph_nblock: Number of
                                      basic blocks in the function
                                      - Gt, graph_transitivity: Transitivity of
                                      the function flow graph
                                      - Gdi, graph_diameter: Diamater of the
                                      function flow graph
                                      - Gnc, graph_num_components: Number of components
                                      in the function (non-connected flow graphs)
                                      - imp, impname: References
                                      to imports in the instruction
                                      - Gp, groups_category: Group of the
                                      instruction (FPU, SSE, stack..)
                                      - Gd, graph_density: Density of the
                                      function flow graph
                                      - cst, cstname: Constant (32/64bits) in the
                                      instruction (not addresses)
                                      - Mt, mnemonic_typed: Mnemonic and type of
                                      operand feature
                                      - Gmd, graph_mean_degree: Mean degree of the function
                                      - dat, datname: References to data in the instruction
                                      - M, mnemonic:
                                      Mnemonic of instructions feature
                                      - Gmib, graph_mean_inst_block: Mean
                                      of instruction per basic blocks in the function  [required]
      -d, --distance <function>       Mathematical distance function between cosine and correlation
                                      [default: auto]
      -t, --threshold FLOAT           Global distance threshold to keep matches between 0.0 to 1.0 [default:
                                      0.00]
      -s, --sparsity FLOAT            Row based sparsity threshold to keep matches between 0.0 to 1.0
                                      [default: 0.20]
      -i, --maxiter INTEGER           Maximum number of iteration for belief propagation [default: 80]
      -tr, --tradeoff FLOAT           Tradeoff betwee callgraph (neat 0.0) and function content (near 1.0)
                                      [default: 0]
      --refine-match / --no-refine-match
      -v, --verbose                   Activate debugging messages
      -h, --help                      Show this message and exit.


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
differ = QBinDiff(p1, p2, distance=0.80, threshold=0.5, maxiter=100)
differ.register_feature(MnemonicSimple())
differ.run(match_refine=True)
matching = differ.matching
```

qBinViz
-------

qBinViz helps vizualizing diffs in IDA as twin views. However using it requires
a python2 version of qBinDiff. The only thing that prevent it to work out of the
box with python2 are types on prototypes. The can be removed with the following
command in the main project directory:

```bash
find . -name '*.py' -exec sed -i -e '/^\s*def/ s/: \?[^,=)]*\([,=)]\)/\1/g' -e '/^\s*def/ s/ -> .*:/:/g' '{}' \;
```


_(We recommend doing it in a separated dedicated branch)_

Then the qbindiff module should be available in the pythonpath so that IDA qBinViz
will find it. A simple solution is to create a symbolic link of qbindiff into $IDA_ROOT/python.
qBinViz can then be triggered as a script with Ctrl+F7 or as a plugin by putting it
in the $IDA_ROOT/plugins directory.