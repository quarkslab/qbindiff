Introduction
============

QBinDiff is an experimental binary-diffing tool addressing the diffing as a `Network Alignement Quadratic Problem <>`_.

> But why developing yet another differ while Bindiff works well?

Bindiff is great, no doubt about it, but we have no control on the diffing process. Also, it works great on standard binaries but it lacks flexibility on some corner-cases (embedded firmwares, diffing two portions of the same binary etc).

A key idea of QBinDiff is enabling tuning the diffing programmatically by:

- writing its own feature
- being able to enforce some matches
- emphasizing either on the content of functions (similarity) or the links between them (callgraph)

In essence, the idea is to be able to diff by defining its own criteria which sometimes, is not the control-flow and instructions but could for instance, be data-oriented.

Last, QBinDiff as primarily been designed with the binary-diffing use-case in mind, but it can be applied to various other use-cases like social-networks. Indeed, diffing two programs boils down to determining the best alignment of the call graph following some similarity criterion.

Indeed, solving this problem is APX-hard, that why QBinDiff uses a machine learning approach (more precisely optimization) to approximate the best match.

Likewise Bindiff, QBinDiff also works using an exported disassembly of program obtained from IDA. Originally using `BinExport <TODO>`_, it now also support `Quokka <TODO>`_ as backend, which extracted files, is more exhaustive and also more compact on disk (good for large binary dataset).

.. warning:: QBinDiff is an experimental tool for power-user where many parameters, features, thresholds or weights can be adjusted. Obtaining good results usually requires tuning these parameters.

.. note:: QBinDiff does not intend to be faster than other differs, but rather being more flexible.

