(howto/qbindiff)=

# Standalone differ

QBinDiff can be used either as a standalone binary differ or as a highly customizable library.
Here we describe how to use it as a standalone binary differ.

Once installed, the executable bash script `qbindiff` is available in the PATH. The command line arguments are the following:

```none
Usage: qbindiff [OPTIONS] <primary file> <secondary file>

  qBinDiff is an experimental binary diffing tool based on machine learning technics, namely Belief propagation.

Options:
  -l, --loader <loader>           Loader type to be used. Must be one of these ['binexport', 'quokka']. [default: binexport]
  -f, --features <feature>        The following features are available:
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
                                    - dat: References to data in the instruction. It's a superset of strref
                                    - wlgk: Weisfeiler-Lehman Graph Kernel. It's strongly suggested to use the cosine distance with this feature. Options: ['max_passes': int]
                                    - fname: Match the function names
                                    - M: Mnemonic of instructions feature
                                    - Mt: Mnemonic and type of operand feature
                                    - Gp: Group of the instruction (FPU, SSE, stack..)
                                    - addr: Address of the function as a feature
                                    - dat: References to data in the instruction. It's a superset of strref
                                    - strref: References to strings in the instruction
                                    - cst: Numeric constant (32/64bits) in the instruction (not addresses)
                                  Features may be weighted by a positive value (default 1.0) and compared with a specificdistance (by default the option -d is used) like this <feature>:<weight>:<distance>
                                  [default: ('wlgk', 'fname', 'addr', 'dat', 'cst')]
  -fopt, --feature-option <feature> <option> <value>
                                  Specify a feature option. To get a list of options accepted by a feature look into the description of the feature
  -n, --normalize                 Normalize the Call Graph (can potentially lead to a partial matching). [default disabled]
  -d, --distance <function>       The following distances are available ('canberra', 'correlation', 'cosine', 'euclidean')
                                  [default: canberra]
  -s, --sparsity-ratio FLOAT      Ratio of least probable matches to ignore. Between 0.0 (nothing is ignored) to 1.0 (only perfect matches are considered) [default: 0.75]
  -sr, --sparse-row               Whether to build the sparse similarity matrix considering its entirety or processing it row per row
  -t, --tradeoff FLOAT            Tradeoff between function content (near 1.0) and call-graph information (near 0.0) [default: 0.75]
  -e, --epsilon FLOAT             Relaxation parameter to enforce convergence [default: 0.50]
  -i, --maxiter INTEGER           Maximum number of iteration for belief propagation [default: 1000]
  -e1, --executable1 PATH         Path to the primary raw executable. Must be provided if using quokka loader
  -e2, --executable2 PATH         Path to the secondary raw executable. Must be provided if using quokka loader
  -o, --output PATH               Write output to PATH
  -ff, --file-format [bindiff]    The file format of the output file. Supported formats are [bindiff]. [default: bindiff]
  --enable-cortexm                Enable the usage of the cortex-m extension when disassembling
  -v, --verbose                   Activate debugging messages. Can be supplied multiple times to increase verbosity
  -h, --help                      Show this message and exit.
```

* `-l`, `--loader` The backend loader that should be used to load the disassembly. The possible values are `binexport, quokka`. The default value is `binexport`.
BinExport is more flexible since it can be used with IDA, Ghidra and BinaryNinja but it generally yields to worst results since it doesn't export as much informations as Quokka and some features don't work with it. Quokka on the other hand usually produces better results but it's only compatible with IDA. If you want to use a different backend loader you can specify a {ref}`custom_backend`
* `-f`, `--features` Specify a single feature that you want to use that will contribute to populate the similarity matrix. You can specify multiple features by using more `-f` arguments. Look at the help page to see the complete list of features. For each feature you must specify the name and optionally a weight and/or a distance that should be used, the syntax is one of the following:
  * `<name>` The default weight is 1.0 and the default distance (`-d`) is used
  * `<name>:<weight>` The weight must be a floating point value > 0. The default distance (`-d`) is used
  * `<name>:<distance>` Look at `-d` for the list of allowed distances. The default weight is 1.0
  * `<name>:<weight>:<distance>`
  
  To know more in detail how the features are combined together to generate a similarity matrix look at {ref}`features`.
* `-fopt`, `--feature-option` Set and option to a feature previously enabled. Not all the features have configurable options. To have a list of the accepted options for a particular feature look into its description
* `-n`, `--normalize` Normalize the Call Graph by removing some of the edges/nodes that should worsen the diffing result. **WARNING:** it can potentially lead to a worse matching. To know the details of the normalization step look at {ref}`normalization`
* `-d`, `--distance` Set the default distance that should be used by the features. The possible values are `canberra, correlation, cosine, euclidean, jaccard-strong`. The default one is `canberra`. To know the details of the jaccard-strong distance look here {ref}`jaccard-strong`
* `-s`, `--sparsity-ratio` Set the density of the similarity matrix. This will loose some information (hence decrease accuracy) but it will also increase the performace. `0.999` means that the 99.9% of the matrix will be filled with zeros. The default value is `0.75`
* `-sr`, `--sparse-row` If this flag is enabled the density value of the sparse similarity matrix will affect each row the matrix. That means that each row of the matrix has the defined sparsity ratio. This guarantees that there won't be rows that are completely erased and filled with zeros.
* `-t`, `--tradeoff` Tradeoff between function content (near 1.0) and call-graph topology information (near 0.0). The default value is `0.75`
* `-e`, `--epsilon` Relaxation parameter to enforce convergence on the belief propagation algorithm. For more information look at {ref}`belief-propagation`. The default value is `0.50`
* `-i`, `--maxiter` Maximum number of iteration for the belief propagation. The default value is `1000`
* `-e1`, `--executable1` Path to the primary raw executable. Must be provided if using the quokka loader, otherwise it is ignored
* `-e2`, `--executable2` Path to the secondary raw executable. Must be provided if using the quokka loader, otherwise it is ignored
* `-o`, `--output` Path to the output file where the result of the diffing is stored
* `-ff`, `--file-format` The file format of the output file. The only supported format for now is `bindiff`. For more information look at {ref}`bindiff` The default value is `bindiff`
* `--enable-cortexm` Enable the usage of the cortex-m extension when disassembling. Only relevant if using the binexport loader.
* `-v`, `--verbose` Increase verbosity. Can be supplied up to 3 times.

```{note}
The less sparse the similarity matrix is the better accuracy we achieve but also the slower the algorithm becomes. In general the right value must be tuned for each use case.

Also note that after a certain threshold reducing the sparsity of the matrix **won't** yield better results.
```

## Examples

```bash
qbindiff -l quokka -e1 binary-primary.exe -e2 binary-secondary.exe \
         binary-primary.qk binary-secondary.qk \
         -f wlgk:cosine \
         -f fname:3 \
         -f dat \
         -f cst \
         -f addr:0.01 \
         -d jaccard-strong -s 0.999 -sr \
         -ff bindiff -o ./result.BinDiff -vv
```

---

```bash
qbindiff binary-primary.BinExport binary-secondary.BinExport \
         -f wlgk:cosine \
         -f fname:3 \
         -f addr:0.01 \
         -s 0.7 \
         -t 0.5
         -ff bindiff -o ./result.BinDiff -vv
```

# Library

Here we describe how to use QBinDiff as a highly customizable library for diffing.
