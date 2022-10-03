# How it works

```{warning}
Since QBinDiff operates both on binaries and generic graphs most of the informations that are herby presented apply in either case, so you should consider the terms "binary" and "graph" as swappable
```

In this section we will explain how QBinDiff internally works, while it is not strictly necessary to know this it might be helpful to better understand how to fine tune the parameters for a specific diffing task.

From a high level prespective QBinDiff operates in the following steps:
1. Normalization of the program Call Graph (**CG**). This step is optional
2. Core of the diffing process. In the end a mapping between the most similar functions is produced. This step is performed in multiple substeps:
   1. Analysis of each function by extracting {ref}`features` and combining them into a similarity matrix
   2. Optionally performing user defined similarity matrix refinements
   3. Combining the similarity matrix and the topology of the CG with a state of the art machine learning algorithm to produce the functions mapping
3. Export the result in different formats

Each step will be described in details
