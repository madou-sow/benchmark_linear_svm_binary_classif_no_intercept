# Comparative Performance Analysis of Optimization Solvers for Linear Support Vector Machines

## Abstract

> **Background :** Support Vector Machines (SVMs) are a fundamental class of supervised learning algorithms. While their theoretical formulation is robust, the efficiency of their numerical implementation depends on the solvers' ability to handle quadratic optimization problems, which are often sensitive to data scaling.

> **Objective :** Comparative evaluation of four solvers using the BenOpt (**Liblinear (Sklearn)**, **Lightning**, **CD**, and **L-BFGS-B**) focusing on their convergence speed, accuracy, and stability with raw or normalized data.

> **Major Result :** The consistent use of the Standard Scaler has a greater impact on performance than choosing between Sklearn and Lightning. The geometry of the data (conditioning of the Hessian) determines the intrinsic difficulty of the optimization problem.

> **Tools Used :** Benchopt Framework is an open-source benchmarking framework for optimization algorithms that guarantees the reproducibility and fairness of comparisons

## Introduction and Context
### Support Vector Machines: Theoretical Foundations

Support Vector Machines (SVMs) are one of the most important classes of supervised learning algorithms in the history of machine learning. Developed in the 1990s based on the theoretical considerations of Vladimir Vapnik on the statistical theory of learning (the Vapnik-Chervonenkis theory) they rely on two fundamental key ideas:

1. The notion of maximum margin: The margin is the distance between the separating hyperplane and the nearest samples (the support vectors). The maximum margin separation boundary has the smallest capacity, ensuring better generalization.

2. **The Kernel Trick**: Projecting data into a higher-dimensional representation space using a kernel function allows handling non-linearly separable cases without explicitly calculating the transformation.

### Binary Classification by Linear SVM

In the case of binary SVM, the algorithm separates two classes by creating an optimal hyperplane in the feature space. The primal optimization problem is written as :

$$\min_{\beta \in \mathbb{R}^p} P(\beta) = \frac{1}{2}\|\beta\|_2^2 + C \sum_{i=1}^{n} \max\left(0, 1 - y_i x_i^\top \beta\right)$$

where :
- $\frac{1}{2}\|\beta\|_2^2$ : regularization term $L_2$ (strict convexity)
- $\max(0, 1 - y_i x_i^\top \beta)$ : **Hinge loss** (convex, but not differentiable in $y_i x_i^\top \beta = 1$)
- $C > 0$ : compromise parameter margin / classification errors

### Problem Statement and Motivation

While the theoretical formulation is robust, its implementation raises important practical questions :

 - Which solver should be chosen based on the data structure ? 
 - What is the impact of normalization on convergence ?
 - How does the parameter $C$ influence the problem's difficulty ?

This study provides **empirical and quantitative** answers to these questions.

## Theoretical Framework and Mathematical Formulation

### Primal and Dual Formulation

The dual problem associated with the linear SVM, obtained using the **Karush-Kuhn-Tucker (KKT)** conditions, is expressed in terms of dual variables $\alpha_i \in [0, C]$ :

$$\max_{\alpha \in \mathbb{R}^n} D(\alpha) = \sum_i \alpha_i - \frac{1}{2} \alpha^\top Q \alpha \quad \text{under duress} 0 \leq \alpha_i \leq C$$

where $Q_{ij} = y_i y_j x_i^\top x_j$ is the **Gram matrix**. Strong duality guarantees $P(\beta^*) = D(\alpha^*)$, and the primal-dual relationship is $\beta^* = \sum_i \alpha_i y_i x_i$.

### Hinge Loss and its Algorithmic Implications

The Hinge Loss $h(u) = \max(0, 1-u)$ is convex but **not differentiable at $u=1$**. This singularity has major consequences:

| Method Type | Impact of Non-Differentiability |
|:---|:---|
| First-order (gradient, CD) | Work on subgradients — **unaffected** |
| Second-order (L-BFGS-B) | The Hessian is zero p.p. and undefined at $u=1$ — **penalized** |

### Conditioning and Geometry

The conditioning of the problem is measured by $\kappa(H)$, the **conditioning number** of the Hessian. A high $\kappa$ implies very elongated isovalue curves (ellipsoids). Normalization transforms:

$$\tilde{x}_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}$$

making the Hessian more **isotropic** ($\kappa \rightarrow 1$) and allowing for rapid convergence in all directions.

### The Four Solvers Studied

| Solver | Family | Implementation | Complexity/iter |
|:---|:---|:---|:---|
| **Sklearn (Liblinear)** | Dual Coordinate Descent | Optimized C++ | $O(\text{nnz}(x_i))$ |
| **Lightning** | Stochastic SDCA | C++/Cython | $O(p)$ on average |
| **CD** | Primal Coordinate Descent | Python/Numba | $O(p)$ |
|**L-BFGS-B** | Quasi-Newton (2nd order) | SciPy (Fortran) | $O(mp)$, $m$ stored vectors |

## Experimental Methodology

### 1. The Benchopt Framework

**Benchopt** is an open-source benchmarking framework for optimization algorithms that guarantees the **reproducibility** and **fairness** of comparisons. It standardizes the interface between solvers and problems, manages dependencies, results caching, and visualization. Benchmarking suite tailored for machine learning workflows. Benchopt enforces a clean separation between problem definitions, data sources, and solver implementations, ensuring
that every algorithm solves the same mathematical problem under identical stopping conditions. The framework automatically handles dependency management, result caching, and interactive visualization of convergence curves.  Benchopt can benchmark machine learning pipelines, including preprocessing, hyperparameters, etc. Benchopt uses accuracy metrics. Benchopt can run on many frameworks: scikit-learn, PyTorch, etc and is actively maintained. Morevoer and
to conclude the main differences between tools, Benchopt is for supervised tasks.

### 2. Installing Benchopt, cloning the benchmark and getting started

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

### 3. Directory Struture 

Here is the content of the directory **benchmark_linear_svm_binary_classif_no_intercept** after running the program benchopt :

```       

|-__cache__
|-__pycache__
|-objective.py
|-test_config.py
|-datasets:
    |-__pycache__
    |-cluster1data5120w512.csv
    |-cluster1data5120w512.py
    |-cluster1data5120w512SS.py
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

### 4.The Heart of the Benchmark

**objective.py:**

Definition: This is the central file that defines the mathematical problem to be solved.

Role: It contains the Objective class, which specifies the loss function (here, the Hinge Loss for the SVM) and the penalty (L2). It also defines how to evaluate the performance of a solver via the evaluate_result method (calculating the primal objective).

### 5.Datasets (Data)

These files provide the X matrices (characteristics) and the y vectors (targets/labels) to the objective.

**datasets/cluster1data5120w512.csv:**

Definition: A raw data file in CSV format.

Role: Contains real-world measurements (CO2, Temperature) associated with clusters. This is the "field" data source.

**datasets/cluster1data5120w512.py:**

Definition: A Python script that inherits from BaseDataset.

Role: Loads the CSV file above, cleans the data, and transforms it into a binary classification problem (e.g., Class 1 vs. the rest) to make it compatible with the SVM objective.

**datasets/libsvm.py:**

Definition: A wrapper for data from the LibSVM library.

Role: Automatically downloads reference datasets (such as news20) from the internet to test solvers on large, sparse datasets.

**datasets/simulated.py:**

Definition: Synthetic data generator.

Role: Creates controlled random data (number of samples, number of variables) to test the behavior of algorithms under specific theoretical conditions.

### 6.Solvers (Optimization Algorithms)

Each file contains a different method for finding the β coefficients that minimize the function defined in the objective.

**solvers/cd.py:** Implements Coordinate Descent (often via the dual). It is an iterative algorithm that optimizes one variable at a time. Uses Numba for speed.

**solvers/l_bfgs_b.py:** Uses the quasi-Newton L-BFGS-B algorithm (via scipy). It uses gradient information to converge faster than simple gradient descent.

**solvers/lightning.py:** Wrapper for the Lightning library. It uses optimized C++ implementations of SDCA or linear SVM solvers.

**solvers/sklearn.py:** Uses the LinearSVC class from scikit-learn (based on the liblinear library). This is often the benchmark.

### 7. Outputs (Results and Analysis)

These files are automatically generated by Benchopt after the `benchopt run` command is executed.

`outputs/*.html` (HTML files):

Definition: Interactive reports.

Purpose: They allow visualization of convergence curves (evolution of the objective function as a function of time or iterations) in a web browser. This allows comparison of which solver is the fastest.

`outputs/*.parquet`:

Definition: Files for storing compressed tabular data.

Purpose: They contain the raw results of each execution (exact time, objective value, parameters, etc.). This format is very efficient for subsequent re-analysis with pandas.

`outputs/cache_run_list.json`:

Definition: Cache index file.

`outputs/cache_run_list.json`:

Definition: Cache index file.

Purpose: Benchopt uses this to determine which combinations (Solver/Dataset/Parameters) have already been executed. This avoids rerunning lengthy calculations if nothing has changed, saving time during subsequent runs.



