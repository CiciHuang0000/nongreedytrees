# Efficient Non-greedy Optimization of Decision Trees

## Overview

Decision trees recently have increased their popularity owing to their computational efficiency and applicability to large-scale classification and regression tasks. Conventional algorithms for decision tree induction are greedy, i.e., a type of heuristic algorithm making the locally optimal choice at each stage. Greedy trees begin with a single node $i$, a split function $s_i$ is optimized based on a subset of training data $D_i$ such that $D_i$ is split into two subsets, which in turn define the training data for the two children of the node i. However, the intrinsic limitation of this procedure is that the optimization of $s_i$ is solely conditioned on $D_i$, i.e., there is no ability to fine-tune the split function $s_i$ based on the results of training at lower levels of the tree. This paper "Efficient Non-greedy Optimization of Decision Trees" proposes a algorithm for optimizing the split functions at all levels of the tree jointly with the leaf parameters based on a global objective that addresses this limitation. 


## Basics of Decision Trees (Greedy and Non-greedy)

## Problem formulation 

This paper focuses on binary classification trees with $m$ split nodes. An input $x \in \mathbb{R}^p$ is directed from the root of the tree. Each internal $m$ node, indexed by $i in {1,...,m}$ performs a binary test by evaluating a node-specific split function $s_i(x): \mathbb{R} \rightarrow {-1, +1}$. That is, if $s_i(x)$ evaluates to -1, then x is directed to the left child of node $i$. Otherwise, $x$ is directed to the right child. Each split function is parameterized by a weight vector $w_i$ which is a linear threshold function $s_i(x) = sgn(\mathbf{w}_i^T \mathbf{x})$. They also incorporated an offset parameter to obtain slit functions $sgn(\mathbf{w}_i^T\mathbf{x}-b_i)$.

After setting up the split function, we get that the conditional probability distribution $p=(y=l|j)$ over class labels $l \in {1,...,k}$ is:
parametrized with a vector of unnormalized predictive log-probabilities,$\mathbf{\theta}_j \in \mathbb{R}^k$ and a softmax function:

$$
p(y=l|j) = \frac{\exp{\mathbf{\theta}_{j[l]}}}{\sum_{\alpha=1}^k \exp{\mathbf{\theta}_{j[\alpha]}}}
$$

where $\mathbf{\theta}_{j[\alphda]}$ denotes the $\alpha_th$ element of vector $\mathbf{\theta}_j$

Therefore, we get our weight matrix to be $W \in \mathbb{R}^{m \times p}$ and unnormalized log-probabilities matrix to be $\Theta \in \mathbb{R}^{(m+1) \times k}$ whose rows comprise weight vectors nd leaf parameters. 

## Algorithm

## Experiments

## Conclusion

## Related work 
