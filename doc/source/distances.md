# Distances

The metrics are used to measure the distance (hence the similarity) between the feature
vectors. Choosing a different distance could lead to different behaviors.
Also note that some features are performing better with a specific distance metric.

Most of the distance functions that can be used are the ones used by scipy, you can
find a list [here](https://docs.scipy.org/doc/scipy/reference/spatial.distance.html).

The ones that are unique in QBinDiff are described below

(jaccard-strong)=

## Jaccard-Strong

This is a experminetal new metric that combines the jaccard index and the canberra
metric.

Formally it is defined as:

$$
d(u, v) = \sum_{i=0}^n\frac{f(u_i, v_i)}{ | \{ i  |  u_i \neq 0 \lor v_i \neq 0 \} | }
$$

$$
with\ u, v \in \mathbb{R}^n
$$
Where the function `f` is defined like this:

$$
f(x, y) = 
\begin{cases}
    0 & \text{if } x = 0 \lor y = 0 \\
    1 - \frac{|x - y|}{|x| + |y|} & \text{otherwise}
\end{cases}
$$
