# qBinDiff

<p align="center">
  <a href="https://github.com/quarkslab/qbindiff/releases">
    <img src="https://img.shields.io/github/v/release/quarkslab/qbindiff?logo=github">
  </a>
  <img src="https://img.shields.io/github/license/quarkslab/qbindiff"/>
  <a href="https://github.com/quarkslab/pastis/releases">
    <img src="https://img.shields.io/github/actions/workflow/status/quarkslab/qbindiff/release.yml">
  </a>
  <img src="https://img.shields.io/github/downloads/quarkslab/tritondse/total"/>
  <img src="https://img.shields.io/pypi/dm/qbindiff"/>
</p>

qBinDiff is an experimental binary diffing addressing the diffing as a **Network Alignement Quadratic Problem**.
But why developing yet another differ when Bindiff works well ?
We love bindiff, but we have no control at all on the diffing process. Also, it works great on standard
binaries but it is more complex to put it in practice on some cornercases (embedded firmwares, diffing
two portions of the same binary etc).

The key idea is to enable **programing the diffing** by:
* writing its own feature
* being able to enforce some matches
* being able to put the emphasis on either the content of functions (similarity)
  or the links between them (callgraph)

In essence, the idea is to be able to diff by defining its own criteria which sometimes, are not the 
control-flow CFG and instruction but more data-oriented for instance.

Last, qbindiff as primarly been designed with the binary-diffing use-case in mind, but
it can be applied to various other use-cases like social-networks. Indeed, diffing two
programs boils down to determining the best alignement of the call graph following some
similarity criterias.

Indeed, solving this problem, APX-hard, that why we use a machine learning approach
(more precisely optimization) to approximate the best match.

Likewise Bindiff, qBinDiff also works using an exported disassembly of program obtained
from IDA. Originally using BinExport, it now also support Quokka as backend which extracted
file is more exhaustive and also more compact on disk (good for large binary dataset).

> Note: qBinDiff is an experimental tool for power-user where many parameters, thresholds
> or weights can be adjusted. Use it at your own risks.

*(Please note that qBinDiff does not intend to be faster to Bindiff or other differ counterparts)*


## Installation

qBinDiff can be installed through pip with:

    pip install qbindiff

As some part of the algorithm are very CPU intensive the installation
will compile some components written in native c.

As depicted above, qBinDiff relies on some projects (also developed at Quarkslab):

* [python-binexport](https://github.com/quarkslab/python-binexport), wrapper on the BinExport protobuf format.
* [python-bindiff](https://github.com/quarkslab/python-bindiff), wrapper around bindiff (used to write results as Bindiff databases)
* [Quokka](https://github.com/quarkslab/quokka), another binary exported based on IDA. Faster than binexport and more exhaustive (thus diffing more relevant)



## Usage (command line)

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


## Library usage

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

## Documentation

The documentation is available on the [diffing portal](https://quarkslab.github.io/diffing-portal/).


## Custom diffing

TODO: Example diffing something unrelated to diffing.


## Papers and conference

TODO:

## Cite qBinDiff

```latex
TODO: ASE
```


## Contributing & Contributors

Any help, or feedback is greatly appreciated via Github issues, pull requests.

**Current**:
* Robin David
* Riccardo Mori
* Roxane Cohen

**Past**:
* Alexis Challande
* Elie Mengin

[**All contributions**](https://github.com/quarkslab/qbindiff/graphs/contributors)
