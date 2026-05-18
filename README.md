# Comparative Performance Analysis of Optimization Solvers for Linear Support Vector Machines

## 📋 Abstract

> **Background :** Support Vector Machines (SVMs) are a fundamental class of supervised learning algorithms. While their theoretical formulation is robust, the efficiency of their numerical implementation depends on the solvers' ability to handle quadratic optimization problems, which are often sensitive to data scaling.

> **Objective :** Comparative evaluation of four solvers using the BenOpt (**Liblinear (Sklearn)**, **Lightning**, **CD**, and **L-BFGS-B**) focusing on their convergence speed, accuracy, and stability with raw or normalized data.

> **Definition :** Benchopt is an open-source benchmarking framework for optimization algorithms that guarantees the reproducibility and fairness of comparisons


## Support Vector Machines: Theoretical Foundations

Support Vector Machines (SVMs) are one of the most important classes of supervised learning algorithms in the history of machine learning. Developed in the 1990s based on the theoretical considerations of Vladimir Vapnik on the statistical theory of learning (the Vapnik-Chervonenkis theory) they rely on two fundamental key ideas:

1. The notion of maximum margin: The margin is the distance between the separating hyperplane and the nearest samples (the support vectors). The maximum margin separation boundary has the smallest capacity, ensuring better generalization.

2. **The Kernel Trick**: Projecting data into a higher-dimensional representation space using a kernel function allows handling non-linearly separable cases without explicitly calculating the transformation.

## 2 Binary Classification by Linear SVM

In the case of binary SVM, the algorithm separates two classes by creating an optimal hyperplane in the feature space. The primal optimization problem is written as :

$$\min_{\beta \in \mathbb{R}^p} P(\beta) = \frac{1}{2}\|\beta\|_2^2 + C \sum_{i=1}^{n} \max\left(0, 1 - y_i x_i^\top \beta\right)$$

where :
- $\frac{1}{2}\|\beta\|_2^2$ : regularization term $L_2$ (strict convexity)
- $\max(0, 1 - y_i x_i^\top \beta)$ : **Hinge loss** (convex, but not differentiable in $y_i x_i^\top \beta = 1$)
- $C > 0$ : compromise parameter margin / classification errors
