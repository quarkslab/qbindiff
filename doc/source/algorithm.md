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

(normalization)=

## Normalization

The normalization of the CG is an optional step that aims at simplifying the CG to produce better results when diffing two binaries.
It simplify the graph by removing thunk functions, i.e. functions that are just trampolines to another function; they usually are just made of a single `JMP` instruction.

Removing thunk functions has the benefit of reducing the size of the binary, hence improving the efficiency and the accuracy.

As a reverser you are usually interested in matching _more interesting_ functions other than thunk functions, that's why you might want to enable the normalization pass.

The normalization pass can be user supplied by subclassing `QBinDiff` and overriding the method `normalize(self, program: Program) -> Program`

(features)=

## Features

_Features_ are heuristic functions that operate at specific level inside the function (operand, instruction, basic block, function) to extract a **feature vector**.
A feature vector is simply a math vector of dimension _n_ whose elements are real numbers.

A feature vector is a convenient way of representing some kind of information that has been extracted from the function that was being analyzed. You can also think of feature vectors as a way of compressing the information that was extracted.

The information that is being extracted should help characterize the function, hence extracting a specific "quality" or "characteristic" of the function.

An example of a feature heuristic can be counting how many basic blocks are there in the function Control Flow Graph (**CFG**), we can arguably say that similar functions should have more or less the same number of blocks and by using that heuristic (or feature) we can give a score on how similar two functions are.
The resulting feature vector should be a vector of dimension 1 in which the only element represents the number of blocks in the function

Obviously the "basic blocks number" feature is useful only when the assumption that similar functions have the same number of blocks is true, if we have two functions that have been compiled with different optimization techniques or that have been obfuscated then the heuristic will produce useless results.

```{note}
Always keep in mind what are the underlying assumption of the features you are using, if they don't hold in your specific context you might end up with bad results
```

Multiple features, combined, distance

## Distances

## Passes

## Similarity matrix

## Belief Propagation

```{warning}
This section requires some math knowledge to be unsderstood
```

## Exporting to BinDiff
