Introduction
============

QBinDiff is an experimental binary diffing addressing the diffing as a Network Alignement Quadratic Problem. But why developing yet another differ when Bindiff works well? We love bindiff, but we have no control at all on the diffing process. Also, it works great on standard binaries but it is more complex to put it in practice on some cornercases (embedded firmwares, diffing two portions of the same binary etc).

The key idea is to enable programing the diffing by:

- writing its own feature
- being able to enforce some matches
- being able to put the emphasis on either the content of functions (similarity) or the links between them (callgraph)

In essence, the idea is to be able to diff by defining its own criteria which sometimes, are not the control-flow CFG and instruction but more data-oriented for instance.

Last, qbindiff as primarly been designed with the binary-diffing use-case in mind, but it can be applied to various other use-cases like social-networks. Indeed, diffing two programs boils down to determining the best alignement of the call graph following some similarity criterias.

Indeed, solving this problem, APX-hard, that why we use a machine learning approach (more precisely optimization) to approximate the best match.

Likewise Bindiff, qBinDiff also works using an exported disassembly of program obtained from IDA. Originally using BinExport, it now also support Quokka as backend which extracted file is more exhaustive and also more compact on disk (good for large binary dataset).

.. note:: qBinDiff is an experimental tool for power-user where many parameters, thresholds or weights can be adjusted. Use it at your own risks.

(Please note that qBinDiff does not intend to be faster to Bindiff or other differ counterparts)