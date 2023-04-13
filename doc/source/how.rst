Binary diffing
==============

Binary diffing aims to automatically compare between two binaries, called primary and secondary, based on their machine code or assembly. 

To start, we recall some definitions.

Control-Flow Graph
------------------

A function may consists of several basic blocks. A basic block is made of instructions that are consecutively executed. A Control-Flow Graph then represents the intraprocedural relation between basic blocks inside a function.


Function Call Graph
-------------------

A Function Call Graph follows the same principle but displays the interprocedural relationships between functions of a program. 

.. image:: images/diff_CG.png

In this image, you can see the difference between the Control-Flow Graph (CG) or the Functin Call Graph or Call Graph (CG). What changes is the level or the granularity of what we look at. 

Method
------

In general, we try to find a one-to-one correspondence between elements of the binaries. The elements may be located at different granularities. Indeed, we usually want to find correspondances at the function level. In other words, we want to match or not each function in the primary versus each function in the secondary. The matching may also happen at the basic block level (but this is more difficult).

QBinDiff operates both on binaries and generic graphs. Indeed, graph matching is not limited to binary diffing only. That is why you should consider the terms "binary" and "graph" as swappable [TODO: add bold or italic].


In this section we will explain how QBinDiff internally works, while it is not strictly necessary to know this it might be helpful to better understand how to fine tune the parameters for a specific diffing task.

From a high level prespective QBinDiff operates in the following steps:
1. Normalization of the program Call Graph (**CG**). This step is optional
2. Core of the diffing process. In the end a mapping between the most similar functions is produced. This step is performed in multiple substeps:
   1. Analysis of each function by extracting {ref}`features` and combining them into a similarity matrix
   2. Optionally performing user defined similarity matrix refinements
   3. Combining the similarity matrix and the topology of the CG with a state of the art machine learning algorithm to produce the functions mapping
3. Export the result in different formats

Each step will be described in details
