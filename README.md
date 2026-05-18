# Comparative Performance Analysis of Optimization Solvers for Linear Support Vector Machines

## 📋 Abstract

> **Background :** Support Vector Machines (SVMs) are a fundamental class of supervised learning algorithms. While their theoretical formulation is robust, the efficiency of their numerical implementation depends on the solvers' ability to handle quadratic optimization problems, which are often sensitive to data scaling.

> **Objective :** Comparative evaluation of four solvers using the BenOpt (**Liblinear (Sklearn)**, **Lightning**, **CD**, and **L-BFGS-B**) focusing on their convergence speed, accuracy, and stability with raw or normalized data.

> **Tools Used :** Benchopt Framework is an open-source benchmarking framework for optimization algorithms that guarantees the reproducibility and fairness of comparisons


## Support Vector Machines: Theoretical Foundations

Support Vector Machines (SVMs) are one of the most important classes of supervised learning algorithms in the history of machine learning. Developed in the 1990s based on the theoretical considerations of Vladimir Vapnik on the statistical theory of learning (the Vapnik-Chervonenkis theory) they rely on two fundamental key ideas:

1. The notion of maximum margin: The margin is the distance between the separating hyperplane and the nearest samples (the support vectors). The maximum margin separation boundary has the smallest capacity, ensuring better generalization.

2. **The Kernel Trick**: Projecting data into a higher-dimensional representation space using a kernel function allows handling non-linearly separable cases without explicitly calculating the transformation.

## Binary Classification by Linear SVM

In the case of binary SVM, the algorithm separates two classes by creating an optimal hyperplane in the feature space. The primal optimization problem is written as :

$$\min_{\beta \in \mathbb{R}^p} P(\beta) = \frac{1}{2}\|\beta\|_2^2 + C \sum_{i=1}^{n} \max\left(0, 1 - y_i x_i^\top \beta\right)$$

where :
- $\frac{1}{2}\|\beta\|_2^2$ : regularization term $L_2$ (strict convexity)
- $\max(0, 1 - y_i x_i^\top \beta)$ : **Hinge loss** (convex, but not differentiable in $y_i x_i^\top \beta = 1$)
- $C > 0$ : compromise parameter margin / classification errors

## Experimental Methodology

### The Benchopt Framework

**Benchopt** is an open-source benchmarking framework for optimization algorithms that guarantees the **reproducibility** and **fairness** of comparisons. It standardizes the interface between solvers and problems, manages dependencies, results caching, and visualization. Benchmarking suite tailored for machine learning workflows. Benchopt enforces a clean separation between problem definitions, data sources, and solver implementations, ensuring
that every algorithm solves the same mathematical problem under identical stopping conditions. The framework automatically handles dependency management, result caching, and interactive visualization of convergence curves.  Benchopt can benchmark machine learning pipelines, including preprocessing, hyperparameters, etc. Benchopt uses accuracy metrics. Benchopt can run on many frameworks: scikit-learn, PyTorch, etc and is actively maintained. Morevoer and
to conclude the main differences between tools, Benchopt is for supervised tasks.

### Installing Benchopt, cloning the benchmark and getting started

It is recommended to use benchopt within a conda environment to fully-benefit from benchopt Command Line Interface (CLI).
run in a terminal

```
## Then run the following command to install the latest release of benchopt
p install -U benchopt

## Clone the benchmark repository and cd to it
git clone https://github.com/madou-sow /benchopt/benchmark_linear_svm_binary_classif_no_intercept

## start by creating a new conda environment and then activate it
conda create -n benchopt python
conda activate benchopt
cd benchmark_linear_svm_binary_classif_no_intercept

## Install the desired solvers automatically with benchopt
benchopt install -s sklearn -s cd -s lightning

# Run on a specific dataset
# benchopt run ./benchmark_linear_svm_binary_classif_no_intercept -d svm_cluster

## Full run (all datasets)
benchopt run ./benchmark_linear_svm_binary_classif_no_intercept

## Run with options can be passed to benchopt run, to restrict the benchmarks to some solvers and/or datasets
benchopt run ./benchmark_linear_svm_binary_classif_no_intercept -s sklearn -d simulated --max-runs 200

```

#### Directory Struture 

Here is the content of the directory **benchmark_linear_svm_binary_classif_no_intercept**:

```       

|-__cache__
|-__pycache__
|-objective.py
|-test_config.py
|-datasets:
    |-__pycache__
    |-cluster1data5120w512.csv
    |-cluster1data5120w512.py
    |-libsvm.py
    |-simulated.py

|-datasets/__pycache__:
            |-cluster1data5120w512.cpython-311.pyc
            |-libsvm.cpython-311.pyc
            |-simulated.cpython-311.pyc

|-outputs:
    |-benchmark_linear_svm_binary_classif_no_intercept.html
    |-benchmark_linear_svm_binary_classif_no_intercept_benchopt_run_2026-04-21_12h26m51.html
    |-benchmark_linear_svm_binary_classif_no_intercept_benchopt_run_2026-04-21_12h44m46.html
    |-benchopt_run_2026-04-21_12h26m51.parquet
    |-benchopt_run_2026-04-21_12h44m46.parquet
    |-cache_run_list.json

|-solvers:
    |-__pycache__
    |-cd.py
    |-l_bfgs_b.py
    |-lightning.py
    |-sklearn.py

|-solvers/__pycache__:
    |-cd.cpython-311.pyc
    |-l_bfgs_b.cpython-311.pyc
    |-lightning.cpython-311.pyc
    |-sklearn.cpython-311.pyc
```
