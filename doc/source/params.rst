Parameters
==========

The differ object [TODO:add link] requires several parameters to compute the matches. These parameters are chosen by the user, which makes QBinDiff highly modular.

Distances
---------

A distance is used to measure the distance (hence the similarity) between the feature vectors. Choosing a different distance could lead to different behaviors.
Note that some features are performing better with a specific distance metric [TODO: which one ? How ?]

Most of the distance functions that QBinDiff uses come from `Scipy <https://docs.scipy.org/doc/scipy/reference/spatial.distance.html>`_. These distances are computed between two 1-D arrays. Candidate distances are then : 
* braycurtis
* canberra
* chebyshev
* cityblock
* correlation
* cosine
* euclidean
* mahalanobis
* minkowski
* seuclidean
* sqeuclidean

However, some distance are unique in QBinDiff, as the jaccard-strong distance. This is a experminetal new metric that combines the jaccard index and the canberra
metric.

Formally it is defined as:

.. math::
   d(u, v) = \sum_{i=0}^n\frac{f(u_i, v_i)}{ | \{ i  |  u_i \neq 0 \lor v_i \neq 0 \} | }

.. math::
   with\ u, v \in \mathbb{R}^n

where the function `f` is defined like this:

.. math::
   f(x, y) = 
   \begin{cases}
    0 & \text{if } x = 0 \lor y = 0 \\
    1 - \frac{|x - y|}{|x| + |y|} & \text{otherwise}
    \end{cases}
    
Epsilon
-------
[TODO]


Tradeoff
--------

QBinDiff relies on two aspects of a binary : either the similarity (between nodes or functions) or the structure provided by Control-Flow Graph or Function-Call-Graph. 

The similarity is computed with a distance over a linear combination of several features. Feature are detailled [TODO:add the link]. These features usually depends on the function attributes. On the contrary, the structure is directly linked on the underlying graphs that come from the binary.

The tradeoff parameter is the weight associated to importance we give to the similarity or the structure. If the tradeoff is equal to 0, then we rely exclusively on the structure to diff our binaries. If the tradeoff is equal to 1, then we rely exclusively on the similarity.

.. warning:: 
	Actually, this is a little bit more complicated than that. 

	Indeed, if you set the tradeoff to 1 but that the features you use to compute the similarity matrix rely in part on the structure (for example, if you choose the feature GraphCommunities [TODO: add link]) then you will nevertheless also consider the structure, even if the tradeoff is 1.

	Similarly, if you set the tradeoff to 0, you also consider the similarity. This is due to a QBinDiff implementation specificity. QBinDiff uses a Belief Propagation (BP) algorithm and the similarity matrix is used to initialize the BP weights. That way, even if you set the tradeoff to 0, the similarity is taken into account. 

	Remember : the tradeoff is the weight you put on the similarity or the structure. But a tradeoff of 0 does not mean you do not consider the similarity at all and the same holds for a tradeoff of 1.


Normalization
-------------

[TODO]

Sparsity
--------
[TODO]

