# Comparative Performance Analysis of Optimization Solvers for Linear Support Vector Machines

## Abstract

> **Background :** Support Vector Machines (SVMs) are a fundamental class of supervised learning algorithms. While their theoretical formulation is robust, the efficiency of their numerical implementation depends on the solvers' ability to handle quadratic optimization problems, which are often sensitive to data scaling.

> **Objective :** Comparative evaluation of four solvers using the BenOpt (**Liblinear (Sklearn)**, **Lightning**, **CD**, and **L-BFGS-B**) focusing on their convergence speed, accuracy, and stability with raw or normalized data.

> **Major Result :** The consistent use of the Standard Scaler has a greater impact on performance than choosing between Sklearn and Lightning. The geometry of the data (conditioning of the Hessian) determines the intrinsic difficulty of the optimization problem.

> **Tools Used :** Benchopt Framework is an open-source benchmarking framework for optimization algorithms that guarantees the reproducibility and fairness of comparisons

## 1. Introduction and Context
### 1.1. Support Vector Machines: Theoretical Foundations

Support Vector Machines (SVMs) are one of the most important classes of supervised learning algorithms in the history of machine learning. Developed in the 1990s based on the theoretical considerations of Vladimir Vapnik on the statistical theory of learning (the Vapnik-Chervonenkis theory) they rely on two fundamental key ideas:

1. The notion of maximum margin: The margin is the distance between the separating hyperplane and the nearest samples (the support vectors). The maximum margin separation boundary has the smallest capacity, ensuring better generalization.

2. **The Kernel Trick**: Projecting data into a higher-dimensional representation space using a kernel function allows handling non-linearly separable cases without explicitly calculating the transformation.

### 1.2. Binary Classification by Linear SVM

In the case of binary SVM, the algorithm separates two classes by creating an optimal hyperplane in the feature space. The primal optimization problem is written as :

$$\min_{\beta \in \mathbb{R}^p} P(\beta) = \frac{1}{2}\|\beta\|_2^2 + C \sum_{i=1}^{n} \max\left(0, 1 - y_i x_i^\top \beta\right)$$

where :
- $\frac{1}{2}\|\beta\|_2^2$ : regularization term $L_2$ (strict convexity)
- $\max(0, 1 - y_i x_i^\top \beta)$ : **Hinge loss** (convex, but not differentiable in $y_i x_i^\top \beta = 1$)
- $C > 0$ : compromise parameter margin / classification errors

### 1.3. Problem Statement and Motivation

While the theoretical formulation is robust, its implementation raises important practical questions :

 - Which solver should be chosen based on the data structure ? 
 - What is the impact of normalization on convergence ?
 - How does the parameter $C$ influence the problem's difficulty ?

This study provides **empirical and quantitative** answers to these questions.

## 2. Theoretical Framework and Mathematical Formulation

### 2.1. Primal and Dual Formulation

The dual problem associated with the linear SVM, obtained using the **Karush-Kuhn-Tucker (KKT)** conditions, is expressed in terms of dual variables $\alpha_i \in [0, C]$ :

$$\max_{\alpha \in \mathbb{R}^n} D(\alpha) = \sum_i \alpha_i - \frac{1}{2} \alpha^\top Q \alpha \quad \text{under duress} 0 \leq \alpha_i \leq C$$

where $Q_{ij} = y_i y_j x_i^\top x_j$ is the **Gram matrix**. Strong duality guarantees $P(\beta^*) = D(\alpha^*)$, and the primal-dual relationship is $\beta^* = \sum_i \alpha_i y_i x_i$.

### 2.2. Hinge Loss and its Algorithmic Implications

The Hinge Loss $h(u) = \max(0, 1-u)$ is convex but **not differentiable at $u=1$**. This singularity has major consequences:

| Method Type | Impact of Non-Differentiability |
|:---|:---|
| First-order (gradient, CD) | Work on subgradients — **unaffected** |
| Second-order (L-BFGS-B) | The Hessian is zero p.p. and undefined at $u=1$ — **penalized** |

### 2.3. Conditioning and Geometry

The conditioning of the problem is measured by $\kappa(H)$, the **conditioning number** of the Hessian. A high $\kappa$ implies very elongated isovalue curves (ellipsoids). Normalization transforms:

$$\tilde{x}_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}$$

making the Hessian more **isotropic** ($\kappa \rightarrow 1$) and allowing for rapid convergence in all directions.

### 2.4. The Four Solvers Studied

| Solver | Family | Implementation | Complexity/iter |
|:---|:---|:---|:---|
| **Sklearn (Liblinear)** | Dual Coordinate Descent | Optimized C++ | $O(\text{nnz}(x_i))$ |
| **Lightning** | Stochastic SDCA | C++/Cython | $O(p)$ on average |
| **CD** | Primal Coordinate Descent | Python/Numba | $O(p)$ |
|**L-BFGS-B** | Quasi-Newton (2nd order) | SciPy (Fortran) | $O(mp)$, $m$ stored vectors |

## 3. Experimental Methodology

### 3.1. The Benchopt Framework

**Benchopt** is an open-source benchmarking framework for optimization algorithms that guarantees the **reproducibility** and **fairness** of comparisons. It standardizes the interface between solvers and problems, manages dependencies, results caching, and visualization. Benchmarking suite tailored for machine learning workflows. Benchopt enforces a clean separation between problem definitions, data sources, and solver implementations, ensuring
that every algorithm solves the same mathematical problem under identical stopping conditions. The framework automatically handles dependency management, result caching, and interactive visualization of convergence curves.  Benchopt can benchmark machine learning pipelines, including preprocessing, hyperparameters, etc. Benchopt uses accuracy metrics. Benchopt can run on many frameworks: scikit-learn, PyTorch, etc and is actively maintained. Morevoer and
to conclude the main differences between tools, Benchopt is for supervised tasks.

### 3.2. Installing Benchopt, cloning the benchmark and getting started

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

### 3.3. Directory Struture 

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

#### 3.3.1. The Heart of the Benchmark

**objective.py:**

Definition: This is the central file that defines the mathematical problem to be solved.

Role: It contains the Objective class, which specifies the loss function (here, the Hinge Loss for the SVM) and the penalty (L2). It also defines how to evaluate the performance of a solver via the evaluate_result method (calculating the primal objective).

#### 3.3.2.Datasets (Data)

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

#### 3.3.3.Solvers (Optimization Algorithms)

Each file contains a different method for finding the β coefficients that minimize the function defined in the objective.

**solvers/cd.py:** Implements Coordinate Descent (often via the dual). It is an iterative algorithm that optimizes one variable at a time. Uses Numba for speed.

**solvers/l_bfgs_b.py:** Uses the quasi-Newton L-BFGS-B algorithm (via scipy). It uses gradient information to converge faster than simple gradient descent.

**solvers/lightning.py:** Wrapper for the Lightning library. It uses optimized C++ implementations of SDCA or linear SVM solvers.

**solvers/sklearn.py:** Uses the LinearSVC class from scikit-learn (based on the liblinear library). This is often the benchmark.

#### 3.3.4. Outputs (Results and Analysis)

These files are automatically generated by Benchopt after the `benchopt run ./benchmark_linear_svm_binary_classif_no_intercept` command is executed.

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

## 4. Experimentation

### 4.1 Objective Function — `objective.py`

The heart of the benchmark is `objective.py`, which defines the primal objective $P(\beta)$ evaluated after each solver iteration:

```python
def evaluate_result(self, beta):
    projected_labels = self.X @ beta
    hinge_loss = np.maximum(1 - self.y * projected_labels, 0.).sum()
    loss = self.C * hinge_loss
    pen = 0.5 * np.dot(beta, beta)
    return loss + pen
```

This function is called identically for every solver, guaranteeing a fair comparison. The parameter $C$ is evaluated at two values: **C = 1.0** (moderate regularization) and **C = 0.1** (strong regularization).

### 4.2 Datasets : Description and Loading

#### 4.2.1 `svm_cluster` — Small Real-World Data

```python
import pandas as pd
import os

# ───────────────────────────────────────────────
# Dataset 1: svm_cluster (real data, p=2)
# ─────────────────────────────────────────────────
# Characteristics: CO2 ∈ [200, 400], T(°C) ∈ [18, 24]
# Problem: Scale imbalance → poor conditioning

# Loading (adjust path if necessary)
csv_path = 'datasets/cluster1data5120w512.csv'
df = pd.read_csv(csv_path)
X_raw = df[['CO2', 'T(°C)']].values
y = np.where(df['Cluster'].values ​​== 1, 1., -1.)

# Simulation of dataset statistics
print("📊 Dataset svm_cluster")
print(f"  n_samples  : ~500")
print(f"  n_features : 2 (CO2, T°C)")
print(f"  CO2        : [200 – 400] ppm")
print(f"  T(°C)      : [18 – 24] °C")
print(f"  ⚠ Scale ratio ≈ 20× → poor packaging")
```
```
>📊 Dataset svm_cluster
  n_samples  : ~500
  n_features : 2 (CO2, T°C)
  CO2        : [200 – 400] ppm
  T(°C)      : [18 – 24] °C
  ⚠ Scale ratio ≈ 20× → poor packaging
``` 

This dataset contains real air-quality measurements from approximately $n = 500$ samples with $p = 2$ features: CO₂ concentration (range 200–400 ppm) and temperature T(°C) (range 18–24°C). The cluster labels are binarized to $\{-1, +1\}$.

The critical characteristic of this dataset is its **scale imbalance**: CO₂ values are roughly 20 times larger than temperature values. This disparity directly translates into a poorly conditioned Hessian with $\kappa(X^\top X) \approx 1216$, as we verify numerically in Section 6.

#### 4.2.2 `libsvm` — Massive Sparse Data

```python
# ─────────────────────────────────────────────
# Dataset 2 : libsvm (news20.binary)
# ─────────────────────────────────────────────
from libsvmdata import fetch_libsvm
X, y = fetch_libsvm('news20.binary')

print("📊 Dataset libsvm (news20.binary)")
print(f" Type: Text Classification (20 Newsgroups)")
print(f" Size: n >> p, high dimension")
print(f" Sparsity: very high (TF-IDF ∈ [0, 1])")
print(f"  ✅ Well packaged → ideal for testing scalability")

```
```
>📊 Dataset libsvm (news20.binary)
 Type: Text Classification (20 Newsgroups)
 Size: n >> p, high dimension
 Sparsity: very high (TF-IDF ∈ [0, 1])
  ✅ Well packaged → ideal for testing scalability
```

The `news20.binary` dataset from the LibSVM repository is a classic text classification benchmark (20 Newsgroups). It features very high dimensionality ($p \gg n$), extreme sparsity (most entries are zero), and well-scaled TF-IDF features in $[0, 1]$. It is ideal for testing solver scalability on the kind of data encountered in natural language processing.

#### 4.2.3 `Simulated` — Synthetic High-Dimensional Data

```python
# ─────────────────────────────────────────────
# Dataset 3 : Simulated (synthetic data)
# ─────────────────────────────────────────────
print("📊 Dataset Simulated")
print(f"  Config 1 : n=300,  p=1000")
print(f"  Config 2 : n=1000, p=300")
print(f"  Generator : benchopt.datasets.make_correlated_data")
print(f"  ✅ Controlled scales → isolates the effect of dimensionality")
```
```
>📊 Dataset Simulated
  Config 1 : n=300,  p=1000
  Config 2 : n=1000, p=300
  Generator : benchopt.datasets.make_correlated_data
  ✅ Controlled scales → isolates the effect of dimensionality
``` 

Generated by Benchopt's `make_correlated_data` function, this dataset provides controlled conditions with two configurations: $(n = 300, p = 1000)$ and $(n = 1000, p = 300)$. Feature scales are balanced by construction, allowing us to isolate the effect of dimensionality and the $n/p$ ratio.

## 5. Experimental Results and Analyses

```python
import pandas as pd

# ─────────────────────────────────────────────
# Summary table of observed results
# ─────────────────────────────────────────────
results = {
    'Dataset / Parameter': [
        'libsvm (C=1.0)',
        'libsvm (C=0.1)',
        'svm_cluster brut (C=1.0)',
        'svm_cluster brut (C=0.1)',
        'svm_cluster_ss (C=1.0)',
        'Simulated n=1000 (C=1.0) — time',
    ],
    'Sklearn': ['~2562.54', '~902.55 ✅', '~817.45 ⚠️', '~80.94', '~287 ✅', '1.64s'],
    'Lightning': ['~2562.54', '~2033.67 ⚠️', '~226.68 ✅', '~82.51', '~287 ✅', '0.24s ⭐'],
    'CD': ['—', '—', 'oscillations ⚠️', '—', '~287 ✅', '1.85s'],
    'L-BFGS-B': ['~2562 (approx)', '—', '—', '—', '~287 ✅', '8.02s'],
}

df_results = pd.DataFrame(results).set_index('Dataset / Parameter')

df_results.style\
    .set_caption("Table 1 — Summary of target values P(β) and convergence time")\
    .set_table_styles([{
        'selector': 'th',
        'props': [('background-color', '#1F3864'), ('color', 'white'),
                  ('font-weight', 'bold'), ('text-align', 'center')]
    }, {
        'selector': 'td',
        'props': [('text-align', 'center'), ('font-family', 'monospace')]
    }, {
        'selector': 'caption',
        'props': [('font-weight', 'bold'), ('font-size', '13px'), ('color', '#1F3864')]
    }])
```
```
                        Table 1 — Summary of target values P(β) and convergence time

                                 Sklearn 	   Lightning 	  CD 	              L-BFGS-B
Dataset/Parameter 	  	  	  	 
libsvm (C=1.0) 	                 ~2562.54 	  ~2562.54 	   — 	               ~2562 (approx)
libsvm (C=0.1) 	                 ~902.55 ✅ 	~2033.67 ⚠️  — 	               —
svm_cluster brut (C=1.0) 	       ~817.45 ⚠️ 	~226.68 ✅ 	 oscillations ⚠️ 	 —
svm_cluster brut (C=0.1) 	       ~80.94 	    ~82.51 	     — 	               —
svm_cluster_ss (C=1.0) 	         ~287 ✅ 	   ~287 ✅ 	    ~287 ✅ 	         ~287 ✅
Simulated n=1000 (C=1.0) — time 	1.64s 	     0.24s ⭐ 	   1.85s 	           8.02s

```

### 5.1 Dataset `libsvm` — Massive and Sparse Data

**Key Observations:**

 - **C = 1.0**: Sklearn (~2562.54, ~33s) and Lightning (~2562.54, ~31.5s) converge to the **same global optimum**. Validation of Lightning's robustness on large datasets.

 - **C = 0.1**: Sklearn reaches the global optimum (~902.55, ~1.9s). **Lightning stops prematurely** at ~2033.67 in 0.7s, with a factor difference of 2.2.

> ⚠️ **Critical Phenomenon**: Lightning's apparent speed masks a **non-convergence**. A weak regularization (C=0.1) makes the landscape flatter in the vicinity of the optimum, requiring more iterations. Lightning satisfies its stopping criterion before reaching the optimum.

### 5.2 Dataset `svm_cluster` — Raw Data (unnormalized)

**Key Observations:**

- **C = 1.0**: Sklearn obtains ~817.45, Lightning ~226.68. **Difference factor 3.6** — Sklearn did not converge.

- **C = 0.1**: Sklearn ~80.94, Lightning ~82.51. Convergence is almost identical (<2% difference).

**Interpretation — The impact of conditioning:**

Poor conditioning of the Hessian is the direct cause. In the dual coordinate descent (DCD) used by Liblinear, the optimal step size for updating $\alpha_i$ is:

$$\Delta\alpha_i^* \propto \frac{1}{\|x_i\|^2}$$

With CO2 values ​​exceeding 300, the steps are extremely small in this direction, forcing Sklearn to stagnate in a region of near-zero gradient.

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Effect of conditioning on the optimization landscape",
             fontsize=14, fontweight='bold', color='#1F3864', y=1.02)

beta = np.linspace(-3, 3, 300)
B1, B2 = np.meshgrid(beta, beta)

# ── Raw data (CO2 ~ 300, T ~ 21)
# Hessian ∝ diag(300², 21²) → very anisotropic
scale_raw = np.array([300.0, 21.0])
Z_raw = 0.5 * ((B1 * scale_raw[0])**2 + (B2 * scale_raw[1])**2) / scale_raw.max()**2

# ── Standardized data (std = 1)
Z_scaled = 0.5 * (B1**2 + B2**2)

for ax, Z, title, color, desc in zip(
    axes,
    [Z_raw, Z_scaled],
    ["Raw data (without StandardScaler)", "Normalized data (with StandardScaler)"],
    ['Reds', 'Blues'],
    ["Very elongated ellipses → slow zigzag convergence",
     "Isotropic circles → direct convergence towards the optimum"]
):
    levels = np.percentile(Z, np.linspace(5, 95, 15))
    cs = ax.contourf(B1, B2, Z, levels=20, cmap=color, alpha=0.7)
    ax.contour(B1, B2, Z, levels=levels, colors='white', linewidths=0.5, alpha=0.5)
    ax.set_title(title, fontweight='bold', color='#1F3864', pad=10)
    ax.set_xlabel(r'$\beta_1$ (CO2)', fontsize=11)
    ax.set_ylabel(r'$\beta_2$ (T°C)', fontsize=11)
    ax.text(0.05, 0.05, desc, transform=ax.transAxes,
            fontsize=9, color='white', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#1F3864', alpha=0.8))
    ax.plot(0, 0, 'w*', markersize=15, label='Optimum β*')
    ax.legend(loc='upper right', fontsize=9)
    plt.colorbar(cs, ax=ax, label="P(β) [normalisé]")

plt.tight_layout()
plt.savefig('~/benchmark_linear_svm_binary_classif_no_intercept/figures/figure1_conditionnement.png', dpi=120, bbox_inches='tight')
plt.show()
print("Figure 1: Optimization landscape — effect of normalization")

``̀`





<img alt="Comparison of Solvers" src="https://github.com/madou-sow/Cloud-Fog-Edge-System-for-Smart-Buildings/blob/figures/figure1_conditionnement.png"  title="Comparison of Solvers"/>
