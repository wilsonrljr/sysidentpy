# SysIdentPy Documentation Restructuring Proposal

## Abstract

This document outlines a reorganization of SysIdentPy’s documentation to improve discoverability, reduce friction for beginners, and align with modern documentation standards. The structure will follow four key categories: **Tutorials**, **How-Tos**, **Explanations**, and **Reference Guide**, with additional sections for contributors and real-world examples.

**Acknowledgments**
This documentation restructuring draws inspiration from [NumPy’s NEP 44](https://numpy.org/neps/nep-0044-restructuring-numpy-docs.html#nep44), adapting its principles of clarity and logical organization to SysIdentPy’s domain-specific needs in system identification and time series forecasting, while emphasizing tutorials and reproducibility.

---

## Motivation and Scope

SysIdentPy’s current documentation (like many scientific Python packages) mixes conceptual explanations, code examples, and API references, which can overwhelm new users. By adopting a user-centric structure inspired by [Diátaxis](https://diataxis.fr/), we aim to:
- Separate learning paths for **beginners** (Tutorials) and **practitioners** (How-Tos).
- Improve material for conceptual understanding (Explanations).
- Maintain a clean, searchable **Reference Guide**.
- Highlight SysIdentPy’s features.

---

## Proposed Structure

### **1. Tutorials**

**Audience:** New users with minimal system identification experience.
**Goal:** Teach foundational workflows with engaging examples.

**Suggested Content:**
- **Absolute Beginner’s Guide**
  - Installing SysIdentPy (with optional dependencies).
  - Your First Model: Fit a NARMAX model to synthetic data.
  - Visualize residuals and model predictions.
- **Domain-Specific Tutorials**
  - Modeling a forced pendulum (mechanical systems).
  - Forecasting energy consumption (time-series analysis).
  - Chaotic systems identification (Lorenz attractor).
- **Integration Tutorials**
  - Using SysIdentPy with scikit-learn pipelines.
  - Hybrid modeling with PyTorch neural networks.

**Format:** Jupyter Notebooks with narrative explanations and code.

---

### **2. How-Tos**

**Audience:** Practitioners solving specific problems.
**Goal:** Provide step-by-step solutions to common tasks.

**Suggested Content:**
- **Model Optimization**
  - Tuning hyperparameters for FROLS.
  - Accelerating model selection with parallel processing.
- **Advanced Workflows**
  - Adding custom basis functions (e.g., wavelet terms).
  - Exporting models for deployment.
- **Debugging**
  - Diagnosing overfitting in nonlinear models.
  - Validating residual whiteness.
- **reproducibility**
  - How to reproduce NARX models from papers.

**Format:** Short, task-focused markdown files with code snippets.

---

### **3. Explanations**
**Audience:** Users seeking rigorous mathematical foundations.
**Goal:** Provide theoretical context for SysIdentPy’s algorithms.

**Content:**
For detailed explanations of system identification theory, nonlinear modeling, and practical case studies, refer to the companion open-source book:

**[Nonlinear System Identification and Forecasting: Theory and Practice with SysIdentPy](https://sysidentpy.org/book/0%20-%20Preface/)**
- Covers NARMAX fundamentals, model structure selection, parameter estimation and validation frameworks.
- Includes real-world case studies (e.g., energy forecasting, mechanical systems).
- Provides code examples using SysIdentPy’s API.

**Topics in the Book:**
- Theoretical derivation of FROLS and MetaMSS.
- Bias-variance tradeoffs in nonlinear system identification and forecasting.
- Hybrid modeling with machine learning.
- etc.

---

### **4. Reference Guide**
**Audience:** Advanced users needing API details.
**Goal:** Comprehensive technical documentation.

**Structure:**
- **Model Structure Selection**
  - `FROLS`, `AOLS`, `MetaMSS`, `ER`.
- **Parameter Estimators**
  - `LeastSquares`, `RecursiveLeastSquares`, `etc.`.
- **Metrics & Validation**
  - `root_relative_squared_error`, `simulation_error`, `etc.`.
- **Basis Functions**
  - `Polynomial`, `Fourier`, `etc.`.
- **Utils**
  - `narmax_tools`, `plot_results`, `etc.`.

**Format:** Auto-generated API docs with cross-linked "See Also" sections.

---

### **5. Developer/Contributor Docs**

**Audience:** Maintainers and open-source contributors.
**Content:**
- **Contributor Guide**
  - Setting up a development environment.
  - Writing tests for new basis functions.
- **Under the Hood**
  - Architecture of the FROLS algorithm.
- **Benchmarking**
  - Reproducibility guidelines for performance tests.

---

### **6. Community & Support**
- **FAQ**: Common installation/modeling issues.
- **Workshops**: Links to conference talks/YouTube tutorials.
- **Citing SysIdentPy**: BibTeX entry and recommended papers.


## Impact
- **Beginners**: Reduced learning curve with guided tutorials.
- **Researchers**: Easier discovery of advanced features (e.g., custom basis functions).
- **Industry Users**: Clear benchmarking guides for model comparisons.

