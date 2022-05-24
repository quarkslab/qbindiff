qBinDiff
========

Experimental binary diffing tool based on the **Network Alignement Quadratic Problem**.

Installation
------------

qBinDiff follows the regular installation process given below:

    pip install .

Usage (command line)
--------------------

After installation, the binary ``qbindiff`` is available in the path.
It takes in input two exported files and start the diffing analysis. The result can then
be exported in a BinDiff file format.
The default format for input files is [BinExport](https://github.com/google/binexport),
for a complete list of backend loader look at the `-l, --loader` option in the help.
The complete command line options are:

    Usage: qbindiff [OPTIONS] <primary file> <secondary file>

      qBinDiff is an experimental binary diffing tool based on machine learning technics, namely Belief propagation.

    Options:
      -l, --loader <loader>         Loader type to be used. Must be one of these ['binexport', 'qbinexport']. [default: binexport]
      -f, --features <feature>      The following features are available:
                                      - bnb: Number of basic blocks in the function
                                      - meanins: Mean number of instructions per basic blocks in the function
                                      - Gmd: Mean degree of the function
                                      - Gd: Density of the function flow graph
                                      - Gnc: Number of components in the function (non-connected flow graphs)
                                      - Gdi: Diamater of the function flow graph
                                      - Gt: Transitivity of the function flow graph
                                      - Gcom: Number of graph communities (Louvain modularity)
                                      - cnb: Number of children of the function
                                      - pnb: Number of parents of the function
                                      - rnb: Number of relatives of the function
                                      - lib: Call to library functions (local function)
                                      - dat: References to data in the instruction
                                      - wlgk: Weisfeiler-Lehman Graph Kernel
                                      - fname: Match the function names
                                      - M: Mnemonic of instructions feature
                                      - Mt: Mnemonic and type of operand feature
                                      - Gp: Group of the instruction (FPU, SSE, stack..)
                                      - addr: Address of the function as a feature
                                      - dat: References to data in the instruction
                                      - cst: Numeric constant (32/64bits) in the instruction (not addresses)
                                    Features may be weighted by a positive value (default 1.0) and compared with a specificdistance (by default the option -d is used) like this <feature>:<weight>:<distance>
                                    [default: ('bnb', 'meanins', 'Gmd', 'Gd', 'Gnc', 'Gdi', 'Gt', 'cnb', 'pnb', 'rnb', 'lib', 'dat', 'M', 'Mt', 'Gp', 'addr', 'dat', 'cst')]
      -n, --normalize               Normalize the Call Graph (can potentially lead to a partial matching). [default disabled]
      -d, --distance <function>     The following distances are available ('canberra', 'correlation', 'cosine', 'euclidean')
                                    [default: canberra]
      -s, --sparsity-ratio FLOAT    Ratio of least probable matches to ignore. Between 0.0 to 1.0 [default: 0.75]
      -t, --tradeoff FLOAT          Tradeoff between function content (near 1.0) and call-graph information (near 0.0) [default: 0.75]
      -e, --epsilon FLOAT           Relaxation parameter to enforce convergence [default: 0.50]
      -i, --maxiter INTEGER         Maximum number of iteration for belief propagation [default: 1000]
      -e1, --executable1 PATH       Path to the primary raw executable. Must be provided if using qbinexport loader
      -e2, --executable2 PATH       Path to the secondary raw executable. Must be provided if using qbinexport loader
      -o, --output PATH             Write output to PATH
      -ff, --file-format [bindiff]  The file format of the output file. Supported formats are [bindiff]. [default: bindiff]
      --enable-cortexm              Enable the usage of the cortex-m extension when disassembling
      -v, --verbose                 Activate debugging messages
      -h, --help                    Show this message and exit.


Usage library
-------------

The strength of qBinDiff is to be usable as a python library. The following snippet shows an example
of loading to binexport files and to compare them using the mnemonic feature.

```python
from qbindiff import QBinDiff, Program
from qbindiff.features import WeisfeilerLehman
from pathlib import Path

p1 = Program(Path("primary.BinExport"))
p2 = Program(Path("secondary.BinExport"))

differ = QBinDiff(p1, p2)
differ.register_feature_extractor(WeisfeilerLehman, 1.0, distance='cosine')

differ.process()

mapping = differ.compute_matching()
output = {(match.primary.addr, match.secondary.addr) for match in mapping}
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

For developers
-------

Since the commit ad057ae9 "Change coding style to `black`" the project switched to the
automated code formatter Black. If you want to retrieve the old, clean `git blame` output
you can either run it with `git blame [file] --ignore-revs-file .git-blame-ignore-revs`
or configure git to automatically ignore that revision with
```bash
$ git config blame.ignoreRevsFile .git-blame-ignore-revs
```

_Please note that GitLab blame interface doesn't support this feature yet (but there's an [open issue](https://gitlab.com/gitlab-org/gitlab/-/issues/31423))_

TODO LIST
---------
If you want to help us improve the tools, here are some items we want to work on. Feel free to contribute

### Functionalities

* adding dependencies & incompatibilities between features
* refactor binexport loader to use: https://gitlab.qb/rdavid/python-binexport
* memory managements (limit the memory used or a toggle)

### Features
