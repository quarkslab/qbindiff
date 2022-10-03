(normalization)=

# Normalization

The normalization of the CG is an optional step that aims at simplifying the CG to produce better results when diffing two binaries.
It simplify the graph by removing thunk functions, i.e. functions that are just trampolines to another function; they usually are just made of a single `JMP` instruction.

Removing thunk functions has the benefit of reducing the size of the binary, hence improving the efficiency and the accuracy.

As a reverser you are usually interested in matching _more interesting_ functions other than thunk functions, that's why you might want to enable the normalization pass.

The normalization pass can be user supplied by subclassing `QBinDiff` and overriding the method `normalize(self, program: Program) -> Program`
