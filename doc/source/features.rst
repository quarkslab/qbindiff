Features
========

Consider two binaries, one *primary* and one *secondary*. *Primary* has :math:`n_{1}` functions, *secondary* has :math:`n_{2}` functions. QBinDiff uses a similarity matrix of shape :math:`(n_{1}, n_{2})` that encodes the similarity between each function of the *primary* and each function of the *secondary*. 

The similarity is computed using different features per function. *Features* are heuristics that operate at specific level inside the function (operand, instruction, basic block, function) to extract a **feature vector**. You can also think of feature vectors as a way of compressing the information that was extracted. This information should help to characterize the function.

An example of a feature heuristic can be counting how many basic blocks are there in the function Control Flow Graph (CFG), we can arguably say that similar functions should have more or less the same number of blocks and by using that heuristic (or feature) we can give a score on how similar two functions are.

QBinDiff provides a lot of features an user can choose to compute the similarity. Indeed, relevant features for diffing is task-dependent and a feature may be very useful for specific diffing and useless in others. For example, the "basic blocks number" feature is useful only when we assume similar functions have the same number of blocks. If our assumption is false, namely, that two functions have been compiled with different optimization techniques or that have been obfuscated, then the heuristic will produce useless results. 

Some of these features comes from or are inspired by other differs, such as `BinDiff <https://www.zynamics.com/bindiff.html>`_ or `Diaphora <https://github.com/joxeankoret/diaphora>`_.

.. warning::
   Always keep in mind what the underlying assumptions of the features you are using are. If they don't hold in your specific context, you might end up with bad results.
  
All the QBinDiff features are listed here, from the larger granularity to the smallest one.

Outside function-level features
-------------------------------

* **ChildNb** : Number of function child of the current function inside the Function Call Graph. 
* **ParentNb** : Number of function parents of the current function inside the Function Call Graph.
* **RelativeNb** : Sum of the features ChildNb and ParentNb
* **LibName** : TODO => test
* **ImpName** :  TODO => test


Inside function-level features
------------------------------

* **BBlockNb** [TODO:add link] : number of basic blocks inside the function. Two functions that come from the same source code but that are compiled differently may have a different number of basic blocks.

* **StronglyConnectedComponents** [TODO:add link] : number of strongly connected components inside the Function Call Graph.

* **BytesHash** [TODO:add link] : Hash of the function, using the instructions sorted by addresses. Two functions that come from the same source code but that are compiled differently will probably have a different hash.

* `Cyclomatic Complexity <https://en.wikipedia.org/wiki/Cyclomatic_complexity>`_

* **MDIndex** [TODO:add link] : this feature is based on previous work of `Dullien et al. <https://www.sto.nato.int/publications/STO%20Meeting%20Proceedings/RTO-MP-IST-091/MP-IST-091-26.pdf>`_. It is slightly modified : indeed, the topological sort is only available for Directed Acyclic Graphs (DAG). 

* **JumpNb** [TODO:add link] : Number of jump in the whole function. 

* **SmallPrimeNumbers** [TODO:add link] : Small-Prime-Number hash, based on mnemonics, based on previous work of `Dullien et al. <https://www.sto.nato.int/publications/STO%20Meeting%20Proceedings/RTO-MP-IST-091/MP-IST-091-26.pdf>`_. It is slightly modified from the theoretical implementation. The modulo is made at each round for computational reasons, not only at the end. [TODO:check the validity of the formula with Z3 (again)]

* **ReadWriteAccess** [TODO:add link] : Number of Read and Write Access to the memory. This feature should be resistant to some obfuscation techniques and compilation options.

* **MaxParentNb** [TODO:add link] : Maximum number of predecessors per basic block in the flowgraph.

* **MaxChildNb** [TODO:add link] : Minimum number of predecessors per basic block in the flowgraph.

* **MaxInsNB** [TODO:add link] : Maximum number of basic blocks instructions in the function.

* **MeanInsNB** [TODO:add link] : Mean number of basic blocks instructions in the function.

* **InstNB** [TODO:add link] : Number of instructions in the whole function.

* **GraphMeanDegree** [TODO:add link] : Mean degree of the function flowgraph.

* **GraphDensity** [TODO:add link] : Density of the function flowgraph.

* **GraphNbComponents** [TODO:add link] : Number of components of the (undirected) function flowgraph. This feature should be almost all the time set to 1. Otherwise, this mean two 

* **GraphDiameter** [TODO:add link] : Diameter of the function flowgraph.

* **GraphTransitivity** [TODO:add link] : Transitivity of the function flowgraph.

* **GraphCommunities** [TODO:add link] : Number of graph communities of the (undirected) function flowgraph.

* **Address** : Address of the function. This feature will not be robust against different compilation options.

* **FuncName** : Name of the function. This feature will not be robust against several obfuscation techniques.

* **WeisfeilerLehman** : This feature applies a Weisfeiler Lehman kernel combined with a Local Sensitive Hashing as labeling function. This feature derives from a `Quarkslab blogpost <https://blog.quarkslab.com/weisfeiler-lehman-graph-kernel-for-binary-function-analysis.html>`_. 

.. warning::
   The WeisfeilerLehman feature suffers from several drawbacks, both theoretically and in term of implementation. It may caused some numerical instability errors depending on your binaries. This feature may be refactor in a near future. 


Basic-block level features
--------------------------

Instruction level features
--------------------------
* **MnemonicSimple**
* **MnemonicTyped**
* **GroupsCategory**
* **DatName** 
* **StrRef**

Operand level features
----------------------

* **Constant**



