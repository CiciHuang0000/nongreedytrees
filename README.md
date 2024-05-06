# Efficient Non-greedy Optimization of Decision Trees

## Overview

Decision trees recently have increased their popularity owing to their computational efficiency and applicability to large-scale classification and regression tasks. Conventional algorithms for decision tree induction are greedy, i.e., a type of heuristic algorithm making the locally optimal choice at each stage. Greedy trees begin with a single node $i$, a split function $s_i$ is optimized based on a subset of training data $D_i$ such that $D_i$ is split into two subsets, which in turn define the training data for the two children of the node i. However, the intrinsic limitation of this procedure is that the optimization of $s_i$ is solely conditioned on $D_i$, i.e., there is no ability to fine-tune the split function $s_i$ based on the results of training at lower levels of the tree. This paper "Efficient Non-greedy Optimization of Decision Trees" proposes a algorithm for optimizing the split functions at all levels of the tree jointly with the leaf parameters based on a global objective that addresses this limitation. 

## Basics of Decision Trees (Greedy and Non-greedy)
Decision trees are hierarchical models used in machine learning for decision-making tasks. They represent decisions and their possible consequences in a tree-like structure, where each internal node represents a decision based on a feature, each branch represents the outcome of that decision, and each leaf node represents a class label or a continuous value. Decision trees are intuitive and easy to interpret, making them valuable for both classification and regression tasks. They recursively partition the input space into subsets based on the features' values, aiming to maximize the homogeneity of the target variable within each subset. This process continues until a stopping criterion is met, resulting in a tree that can be used to make predictions on new data by following the path from the root to a leaf node. Additionally, decision trees can handle both categorical and numerical data, making them a suitable tool for various applications. Decision trees and forests are widely utilized in machine learning and have gained increased popularity due to their computational efficiency and suitability for large-scale classification and regression tasks.

Greedy algorithms for decision tree construction optimize split functions one node at a time based on predefined criteria. However, this strategy often results in suboptimal trees. In this paper, the research introduces an algorithm designed to address the inherent limitation of such greedy algorithms, wherein the split function $s_i$ lacks refinement based on outcomes from training at lower levels of the tree. The proposed algorithm guided by a global objective optimizes split functions across all levels of the tree. The paper demonstrates that the task of finding optimal linear-combination splits for decision trees is similar to structured prediction with latent variables, and proposes a convex-concave upper bound on the tree's empirical loss. This bound acts as a surrogate objective, optimized via stochastic gradient descent (SGD) to determine a locally optimal arrangement of the split functions. While conventional algorithms for decision tree induction remain greedy, the paper’s proposed framework offers a non-greedy approach to learning split parameters, addressing some limitations of greedy approaches. By jointly optimizing split functions across different levels of the tree, their algorithm promotes collaboration among split nodes, resulting in more concise trees and improved generalization performance.

** insert video here**

## Problem formulation 

The paper focuses on binary classification trees as structures with internal nodes (nodes with children) that direct data to one of the leaf nodes (nodes without children). Each internal node performs a binary test using a **split function**, determining whether to send the data point to the left or right child node. The leaf nodes then classify the data point based on **predetermined parameters**.

**Tree Structure Set Up**

The tree has $m$ internal nodes and $m+1$ leaf nodes. An input $x \in \mathbb{R}^p$ is directed from the root of the tree. Each internal $m$ node, indexed by 
$i \in {1,...,m}$ performs a binary test by evaluating a node-specific split function $s_i(x): \mathbb{R} \rightarrow {-1, +1}$. That is, if $s_i(x)$ evaluates to -1, then x is directed to the left child of node $i$. Otherwise, $x$ is directed to the right child. Split function use linear threshold functions and is defined as $sgn(\mathbf{w}_i^T\mathbf{x}-b_i)$, where $w_i$ is a weight vector.

Then each leaf node, indexed by $j \in {1,...,m+1}$, specifies a conditional probability distribution $p(y=l|j)$ over class labels $l \in {1,...,k}$ is:
parametrized with a vector of unnormalized predictive log-probabilities, $\mathbf{\theta}_j \in \mathbb{R} ^k$ and a softmax function:

$$p(y = l| j) = \frac{\exp \{\theta_{j[l]}}}{\sum_{\alpha = 1}^{k}  exp{ \theta_{j[\alpha]}}}$$

where $\theta_{j[\alpha]}$ denotes the $\alpha^{th}$ element of vector $\mathbf{\theta}_j$

Therefore, we get our weight matrix for the entire tree to be $W \in \mathbb{R}^{m \times p}$ and unnormalized log-probabilities matrix to be $\Theta \in \mathbb{R}^{(m+1) \times k}$ whose rows comprise weight vectors and leaf parameters. 

**Optimization Goal**

Given a dataset of input-output pairs, $D: {x_z,y_z}_{z=1}^n$, where $y_z \in {1,...,k}$ is the truth class label associated with input $x$. Here, we wish to find $W$ and $\Theta$ that minimize misclassification loss on the training set. However, the joint optimization of the split functions and leaf parameters is known to be extremely challenging due to discrete, sequential nature of decisions in trees.

**Latent Structured Prediction**

So how can evaluate the loss? One key idea here is to establish a link between the decision tree optimization problem and the problem of structured prediction with **latent variables**. One can evaluate all of the split functions for every internal node of the tree on input $x$ by computing $sgn(W\mathbf{x})$ where $sgn(\cdot)$ is the element-wise sign function. Therefore, we can link decision trees learning to latent structured prediction by thinking an m-bit vector of potential split decisions, e.g., $\mathbf{h}=sgn(Wx) \in {-1,+1}^m$, as a latent variable. The latent variable then determines the leaf to which a data point is directed, and then classified using the leaf parameters. To formulate the loss, we introduce a tree navigation function $f: \mathbb{H}^m \rightarrow \mathbb{I}_(m+1)$ that maps an m-bit sequence of split decisions to an indicator vector that specifies a 1-of-(m+1) encoding. That is, each internal node can be represented by this binary indicator vector, forming a latent variable that defines the path to a specific leaf node. The classification loss is computed based on the final leaf node reached.

**Challenge in Directly Optimizing Empirical Loss**

Using the notations, we define $\mathbf{\theta}=\Theta^T f(sgn(Wx)$ to represent the parameters corresponding to the leaf to which x is directed by th split functions in W. A natural choice for the loss function would be the squared loss in regression, which is defined as $l(\mathbf{\theta},y)=|\mathbf{\theta}-y|^2$. Given a training set $D$, we try to minimize 

$$
L(W, \Theta, D)= \sum_{(x,y) \in D} l(\Theta^T f (sgn(Wx)),y)
$$

However, direct global optimization of this empirical loss is challenging. 


## Algorithm

## Experiments

## Conclusion

## Related work 
