# QBinDiff

<p align="center">
  <a href="https://github.com/quarkslab/qbindiff/releases">
    <img src="https://img.shields.io/github/v/release/quarkslab/qbindiff?logo=github">
  </a>
  <img src="https://img.shields.io/github/license/quarkslab/qbindiff"/>
  <a href="https://github.com/quarkslab/qbindiff/releases">
    <img src="https://img.shields.io/github/actions/workflow/status/quarkslab/qbindiff/release.yml">
  </a>
  <img src="https://img.shields.io/github/downloads/quarkslab/qbindiff/total"/>
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

> [!NOTE]
> QBinDiff is an experimental tool for power-user where many parameters, features, thresholds
> or weights can be adjusted. Obtaining good results usually requires tuning these parameters.

*(Please note that QBinDiff does not intend to be faster than other differs, but rather being more flexible.)*

> [!WARNING]
> QBinDiff graph alignment is very memory intensive (compute large matrices), it can fill RAM if not cautious. 
> Try not diffing binaries larger than +10k functions. For large program use very high sparsity ratio (0.99). 

## Documentation

The documentation can be found on the [diffing portal](https://diffing.quarkslab.com/qbindiff/doc/source/intro.html)
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
for a complete list of backend loader look at the `-l1, --primary-loader` option in the help.
The complete command line options are:


```commandline
 Usage: qbindiff [OPTIONS] <primary file> <secondary file>                                                                                                                                    
                                                                                                                                                                                              
 QBinDiff is an experimental binary diffing tool based on machine learning technics, namely Belief propagation.                                                                               
 Examples:                                                                                                                                                                                    
 - For Quokka exports: qbindiff -e1 file1.bin -e2 file2.bin file1.quokka file2.quokka                                                                                                         
 - For BinExport exports, changing the output path: qbindiff -o my_diff.bindiff file1.BinExport file2.BinExport                                                                               
                                                                                                                                                                                              
╭─ Output parameters ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 --output  -o   Output file path. (FILE) [default: qbindiff_results.csv]                                                                                                
 --format  -ff  Output file format. (bindiff|csv) [default: csv]                                                                                                        
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
╭─ Primary file options ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 --primary-loader      -l1  Enforce loader type. (binexport|quokka|ida)                                                                                                
 --primary-executable  -e1  Path to the raw executable (required for quokka exports). (PATH)                                                                           
 --primary-arch        -a1  Enforce disassembling architecture. Format is like 'CS_ARCH_X86:CS_MODE_64'. (TEXT)                                                        
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
╭─ Secondary file options ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 --secondary-loader      -l2  Enforce loader type. (binexport|quokka|ida)                                                                                              
 --secondary-executable  -e2  Path to the raw executable (required for quokka exports). (PATH)                                                                         
 --secondary-arch        -a2  Enforce disassembling architecture. Format is like 'CS_ARCH_X86:CS_MODE_64'. (TEXT)                                                      
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
╭─ Global options ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 --verbose  -v  Activate debugging messages. (-v|-vv|-vvv)                                                                                                             
 --quiet    -q  Do not display progress bars and final statistics.                                                                                                     
 --help     -h  Show this message and exit.                                                                                                                            
 --version      Show the version and exit.                                                                                                                             
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
╭─ Diffing parameters ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 --feature         -f   Features to use for the binary analysis, it can be specified multiple times.                  (<feature>)                                      
                        Features may be weighted by a positive value (default 1.0) and/or compared with a                                                              
                        specific distance (by default the option -d is used) like this <feature>:<weight>:<distance>.                                                  
                        For a list of all the features available see --list-features.                                                                                  
 --list-features        List all the available features.                                                                                                               
 --normalize       -n   Normalize the Call Graph (can potentially lead to a partial matching).                                                                         
 --distance        -d   Available distances: (canberra|euclidean|cosine|haussmann) [default: haussmann]                                                                
 --tradeoff        -t   Tradeoff between function content (near 1.0) and call-graph information (near 0.0). (FLOAT) [default: 0.8]                                     
 --sparsity-ratio  -s   Ratio of least probable matches to ignore. Between 0.0 (nothing is ignored) to 1.0 (only perfect matches are considered) (FLOAT) [default: 0.6]
 --sparse-row      -sr  Whether to build the sparse similarity matrix considering its entirety or processing it row per row.                                           
 --epsilon         -e   Relaxation parameter to enforce convergence. (FLOAT) [default: 0.9]                                                                            
 --maxiter         -i   Maximum number of iteration for belief propagation. (INTEGER) [default: 1000]                                                                  
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
╭─ Passes parameters ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
│ --pass-feature-hash    Anchor matches when function have the same feature hash.                                                                                       
│ --pass-user-defined    Anchor matches using user defined matches. Format is like 'primary-addr1:secondary-addr2,...'. (TEXT)                                          
│ --pass-flirt-hash      Anchor matches using FLIRT/FunctionID like signatures.                                                                                         
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```

### Quokka example

Quokka exporter needs the path of the executable file so one should also use ``-e1`` and ``-e2``.

    $ qbindiff -e1 primary.exe -e2 secondary.exe primary.exe.Quokka secondary.exe.Quokka

Note that we use default values for all parameters, but one can configure the different
features used and the various parameters used.

    $ qbindiff -e1 primary.exe \
               -e2 secondary.exe \
               -f bnb \             # basic block number
               -f cc:3.0 \          # cyclomatic complexity feature
               -f cst:5.0 \         # feature based on constants
               --maxiter 100 \      # maximum number of iterations 
               primary.exe.Quokka \
               secondary.exe.Quokka

### BinExport example

The most simple example generating a diff file in a ``.BinDiff`` format is:

    $ qbindiff primary.BinExport secondary.BinExport -ff bindiff -o out.BinDiff

Binexport backend used, also relies on capstone for the disassembly of instructions and some features.
Thus for some architecture and especially ARM/thumb mode, one should provide the exact disassembly mode
using capstone naming for [architecture identifier](https://github.com/capstone-engine/capstone/blob/f81eb3affaa04a66411af12cf75522cb9649cf83/bindings/python/capstone/__init__.py#L207) and [mode identifiers](https://github.com/capstone-engine/capstone/blob/f81eb3affaa04a66411af12cf75522cb9649cf83/bindings/python/capstone/__init__.py#L231). Thus to diff two binexport files and specifying the exact architecture one can do:

    $ qbindiff primary.BinExport secondary.BinExport -a1 CS_ARCH_ARM:CS_MODE_THUMB -a2 CS_ARCH_ARM:CS_MODE_THUMB


## Library usage

The strength of qBinDiff is to be usable as a python library. The following snippet shows an example
of loading to binexport files and to compare them using the mnemonic feature.

```python
from qbindiff import QBinDiff, Program
from qbindiff.features import MnemonicTyped
from pathlib import Path

p1 = Program("primary.BinExport")
p2 = Program("secondary.BinExport")

differ = QBinDiff(p1, p2)
differ.register_feature_extractor(MnemonicTyped, 1.0)
# Add other features if you want to

differ.process()

mapping = differ.compute_matching()
output = {(match.primary.addr, match.secondary.addr) for match in mapping}
```
## Citing this work
If you use QBinDiff in your work, please consider to cite it using these references : 

```
@inproceedings{CAIDQBinDiff,
  author    = "Cohen, Roxane and David, Robin and Mori, Riccardo and Yger, Florian and Rossi, Fabrice",
  title     = "Improving binary diffing through similarity and matching intricacies",
  booktitle = "Proc. of the 6th Conference on Artificial Intelligence for Defense",
  year      = 2024,
}
```

```
@misc{SSTICQBinDiff,
  title        = "QBinDiff: A modular differ to enhance binary diffing and graph alignment",
  author       = "Cohen, Roxane and David, Robin and Mori, Riccardo and Yger, Florian and Rossi, Fabrice",
  howpublished = "\url{https://www.sstic.org/2024/presentation/qbindiff_a_modular_differ/}",
  year         = 2024,
}
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
