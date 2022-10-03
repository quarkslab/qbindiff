(features)=

# Features

**A detailed list of all the features [here](feature_list)**

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

(feature_list)=

## Feature list

TODO
Multiple features, combined, distance
