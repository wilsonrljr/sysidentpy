---
title: 'SysIdentPy: A Python package for System Identification'
tags:
  - Python
  - System Identification
  - NARMAX
  - Dynamical Systems
  - Model Structure Selection
authors:
  - name: Wilson Rocha Lacerda Junior
    orcid: 0000-0002-3263-1152
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Luan Pascoal da Costa Andrade
    affiliation: 2
  - name: Samuel Carlos Pessoa Oliveira
    affiliation: 2
  - name: Samir Angelo Milani Martins
    affiliation: "1, 2"
affiliations:
 - name: GCoM - Modeling and Control Group at Federal University of São João del-Rei
   index: 1
 - name: Department of Electrical Engineering at Federal University of São João del-Rei
   index: 2
date: 18 May 2020
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

The field of System Identification (SI) aims to build mathematical models for static and dynamic behavior from experimental data[@Lju1987]. In particular, nonlinear system identification has become a central issue in SI community and from the $1950$s onwards many methods has been proposed. In this respect, NARMAX (Nonlinear AutoRegressive Moving Average with eXogenous input) models are among the most well-documented and used model representation of dynamical systems[@Bil2013].

The NARMAX model was proposed by [@BL1981; @LB1985; @CB1989] and can be described as

\begin{equation}
y_k= F^\ell[y_{k-1}, \dotsc, y_{k-n_y},x_{k-d}, x_{k-d-1}, \dotsc, x_{k-d-n_x} + e_{k-1}, \dotsc, e_{k-n_e}] + e_k,
\end{equation}

where $n_y\in \mathbb{N}^*$, $n_x \in \mathbb{N}$, $n_e \in \mathbb{N}$ , are the maximum lags for the system output and input respectively; $x_k \in \mathbb{R}^{n_x}$ is the system input and $y_k \in \mathbb{R}^{n_y}$ is the system output at discrete time~$k \in \mathbb{N}^n$; $e_k \in \mathbb{R}^{n_e}$ represents uncertainties and possible noise at discrete time~$k$. In this case, $\mathcal{F}^\ell$ is some nonlinear function of the input and output regressors with nonlinearity degree $\ell \in \mathbb{N}$ and $d$ is a time delay typically set to $d=1$.

Although there are many possible approximations of $\mathcal{F}(\cdot)$ (e.g., Neural Networks, Fuzzy, Wavelet, Radial Basis Function), the power-form Polynomial NARMAX model is the most commonly used[@Bil2013; @MA2016]:

\begin{align}
\label{eq5:narx}
y_k =& \sum_{0} + \sum_{i=1}^{p}\Theta_{y}^{i}y_{k-i} + \sum_{j=1}^{q}\Theta_{e}^{j}e_{k-j} + \sum_{m=1}^{r}\Theta_{x}^{m}x_{k-m} \nonumber \\
& + \sum_{i=1}^{p}\sum_{j=1}^{q}\Theta_{ye}^{ij}y_{k-i} e_{k-j} + \sum_{i=1}^{p}\sum_{m=1}^{r}\Theta_{yx}^{im}y_{k-i} x_{k-m}  \nonumber \\
& + \sum_{j=1}^{q}\sum_{m=1}^{r}\Theta_{e x}^{jm}e_{k-j} x_{k-m}  \nonumber \\
&  + \sum_{i=1}^{p}\sum_{j=1}^{q}\sum_{m=1}^{r}\Theta_{y e x}^{ijm}y_{k-i} e_{k-j} x_{k-m} \nonumber \\
& + \sum_{m_1=1}^{r} \sum_{m_2=m_1}^{r}\Theta_{x^2}^{m_1 m_2} x_{k-m_1} x_{k-m_2} \dotsc \nonumber \\
& + \sum_{m_1=1}^{r} \dotsc \sum_{m_l=m_{l-1}}^{r} \Theta_{x^l}^{m_1, \dotsc, m_2} x_{k-m_1} x_{k-m_l},
\end{align}

Polynomial NARMAX models have many interesting atrributes[@Bil2013; @Agu2004]:

- All polynomial functions are smooth in $\mathbb{R}$;
- The Weierstrass theorem [@Wei1885] states that any given continuous real-valued function defined on a closed and bounded space $[a,b]$ can be uniformly approximated using a polynomial on that interval;
- Can describe several nonlinear dynamical systems, which include industrial processes, control systems, structural systems, economic and financial systems, biology, medicine, social systems, and much more[@WMNL2019; @FWHM2003; @GGBW2016; @KGHK2003; @BBWL2018; CER2001; @Bil2013; @Agu2004];
- Polynomial NARMAX models can be used both for prediction and inference.

# SysIdentPy

SysIdentPy is an open source python package for system identification using polynomial NARMAX models. The package can handle SISO and MISO NARMAX models identification. It provides various tools including classical Model Structure Selection (MSS) algorithms (e.g., Forward Regression Orthogonal Least Squares); parameter estimation using ordinary Least Squares, recursive algorithms and adaptative filters; four different information criterion methods for order selection (AIC, BIC, LILC, FPE); regression metrics; residual analysis  and so on. The reader is referred to the package documentation for further details.

SysIdentpy is designed designed to be easily expanded and user friendly. Moreover, the package aims to provid useful tools for researchers and students not only in SI field, but also in correlated areas such as Machine Learning, Statistical Learning and Data Science.

The table below and \autoref{fig:example} serves as an example of one of the outputs of the package.

| Regressors     | Parameters | ERR        |
|----------------|------------|------------|
| x2(k-1)        | 0.6000     | 0.90482955 |
| x2(k-2)x1(k-1) | -0.3000    | 0.05072675 |
| y(k-1)^2       | 0.3999     | 0.04410386 |
| x1(k-1)y(k-1)  | 0.1000     | 0.00033239 |


![Caption for example figure.\label{fig:example}](output_16_0.png)

# Future

Future releases will include new methods for MSS of polynomial NARMAX models, multiobjective MSS and parameter estimation algorithms, new adaptative filters, frequency domain analysis, and algorithms for using NARMAX models for classification problems.

# References