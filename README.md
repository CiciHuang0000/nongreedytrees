# Efficient Non-greedy Optimization of Decision Trees

## Overview

Decision trees recently have increased their popularity owing to their computational efficiency and applicability to large-scale classification and regression tasks. Conventional algorithms for decision tree induction are greedy, i.e., a type of heuristic algorithm making the locally optimal choice at each stage. Greedy trees begin with a single node $i$, a split function $s_i$ is optimized based on a subset of training data $D_i$ such that $D_i$ is split into two subsets, which in turn define the training data for the two children of the node i. However, the limitation of this procedure is that the optimization of $s_i$ is solely conditioned on $D_i$, i.e., there is no ability to fine-tune the split function $s_i$ based on the results of training at lower levels of the tree. This paper "Efficient Non-greedy Optimization of Decision Trees" proposes a algorithm for optimizing the split functions at all levels of the tree jointly with the leaf parameters based on a global objective that addresses this limitation. 


## Basics of Decision Trees (Greedy and Non-greedy)

## Problem formulation 

## Algorithm

## Experiments

## Conclusion

## Related work 
