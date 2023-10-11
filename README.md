# QBinDiff

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

QBinDiff is an experimental binary diffing tool addressing the diffing as a **Network Alignement Quadratic Problem**.

> But why developing yet another differ when Bindiff works well?

Bindiff is great, no doubt about it, but we have no control on the diffing process. Also, it works
great on standard binaries but it lacks flexibility on some corner-cases (embedded firmwares,
diffing two portions of the same binary etc...).

A key idea of QBinDiff is enabling tuning the diffing **programmatically** by:
* writing its own feature
* being able to enforce some matches
* emphasizing either on the content of functions (similarity) or the links between them (callgraph)

In essence, the idea is to be able to diff by defining its own criteria which sometimes, is not the
control-flow and instructions but could for instance, be data-oriented.

Last, QBinDiff as primarily been designed with the binary-diffing use-case in mind, but it can be
applied to various other use-cases like social-networks. Indeed, diffing two programs boils down to
determining the best alignment of the call graph following some similarity criterion.

Indeed, solving this problem is APX-hard, that why QBinDiff uses a machine learning approach (more
precisely optimization) to approximate the best match.

Like Bindiff, QBinDiff also works using an exported disassembly of program obtained from IDA.
Originally using [BinExport](https://github.com/google/binexport), it now also support
[Quokka](https://github.com/quarkslab/quokka) as backend, which extracted files, are
more exhaustive and also more compact on disk (good for large binary dataset).

> Note: QBinDiff is an experimental tool for power-user where many parameters, features, thresholds
> or weights can be adjusted. Obtaining good results usually requires tuning these parameters.

*(Please note that QBinDiff does not intend to be faster than other differs, but rather being more flexible.)*


## Documentation

The documentation can be found on the [diffing portal](https://diffing.quarkslab.com/qbindiff)
or can be manually built with

    pip install .[doc]
    cd doc
    make html

Below you will find some sections extracted from the documentation. Please refer to the full
documentation in case of issues.

## Installation

QBinDiff can be installed through pip with:

    pip install qbindiff

As some part of the algorithm are very CPU intensive the installation
will compile some components written in native C/C++.

As depicted above, QBinDiff relies on some projects (also developed at Quarkslab):

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
