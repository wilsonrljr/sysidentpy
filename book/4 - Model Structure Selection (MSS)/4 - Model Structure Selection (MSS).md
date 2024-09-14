## Introduction

> This section is taken mainly from my master thesis, which was based on [Billings, S. A](https://www.wiley.com/en-us/Nonlinear+System+Identification%3A+NARMAX+Methods+in+the+Time%2C+Frequency%2C+and+Spatio-Temporal+Domains-p-9781119943594).

Selecting the model structure is crucial to develop models that can correctly reproduce the system behavior. If some prior information about the system are known, e.g., the dynamic order and degree of nonlinearity, determining the terms and then estimate the parameters is trivial. In real life scenarios, however, in most of the times there is no information about what terms should be included in the model and the correct regressors has to be selected in the identification framework. If the MSS is not performed with the necessary concerns, the scientific law that describes the system may will not be revealed and resulting in misleading interpretations about the system. To illustrate this scenario, consider the following example.

Let $\mathcal{D}$ denote an arbitrary dataset

$$
\begin{equation}
    \mathcal{D} = \{(x_k, y_k), k = 1, 2, \dotsc, n\},
\end{equation}
\tag{1}
$$

where $x_k \in \mathbb{R}^{n_x}$ and $y_k\in \mathbb{R}^{n_y}$ are the input and output of an unknown system and $n$ is the number of samples in the dataset. The following are two polynomial NARX models built to describe that system:

$$
\begin{align}
    y_{ak} &= 0.7077y_{ak-1} + 0.1642u_{k-1} + 0.1280u_{k-2}
\end{align}
\tag{2}
$$

$$
\begin{align}
    y_{bk} &= 0.7103y_{bk-1} + 0.1458u_{k-1} + 0.1631u_{k-2} \\
           &\quad - 1467y_{bk-1}^3 + 0.0710y_{bk-2}^3 + 0.0554y_{bk-3}^2u_{k-3}.
\end{align}
\tag{3}
$$

Figure 1 shows the predicted values of each model and the real data. As can be observed, the nonlinear model 2 seems to fit the data better than the linear model 1. The original system under consideration is an RLC circuit, consisting of a resistor (R), inductor (L), and capacitor (C) connected in series with a voltage source. It is well known that the behavior of such an RLC series circuit can be accurately described by a linear second-order differential equation that relates the current $I(t)$ and the applied voltage $V(t)$:

$$
L\frac{d^2I(t)}{dt^2} + R\frac{dI(t)}{dt} + \frac{1}{C}I(t) = \frac{dV(t)}{dt}
\tag{4}
$$

Given this linear relationship, an adequate model for the RLC circuit should reflect this second-order linearity. While Model 2, which includes nonlinear terms, may provide a closer fit to the data, it is clearly over-parameterized. Such over-parameterization can introduce spurious nonlinear effects, often referred to as "ghost" nonlinearities, which do not correspond to the actual dynamics of the system. Therefore, these models need to be interpreted with caution, as the use of an overly complex model could obscure the true linear nature of the system and lead to incorrect conclusions about its behavior.

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/rlc.png?raw=true)
> Figure 1.Results for two polynomial NARX models fitted to data from an unknown system. Model 1 (left) is a linear model, while Model 2 (right) includes nonlinear terms. The figure illustrates that Model 2 provides a closer fit to the data compared to Model 1. However, since the original system is a linear RLC circuit known to have a second-order linear behavior, the improved fit of Model 2 may be misleading due to over-parameterization. This highlights the importance of considering the physical characteristics of the system when interpreting model results to avoid misinterpretation of artificial nonlinearities. Reference: [Meta Model Structure Selection: An Algorithm For Building Polynomial NARX Models For Regression And Classification](https://ufsj.edu.br/portal2-repositorio/File/ppgel/225-2020-02-17-DissertacaoWilsonLacerda.pdf)

Correctly identifying the structure of a model is crucial for accurately analyzing the system's dynamics. A well-chosen model structure ensures that the model reflects the true behavior of the system, allowing for consistent and meaningful analysis. In this respect, several algorithms have been developed to select the appropriate terms for constructing a polynomial NARX model. The primary goal of model structure selection (MSS) algorithms is to reveal the system's characteristics by producing the simplest model that adequately describes the data. While some systems may indeed require more complex models, it is essential to strike a balance between simplicity and accuracy. As Einstein aptly put it:

> A model should be as simple as possible, but not simpler.

This principle emphasizes the importance of avoiding unnecessary complexity while ensuring that the model still captures the essential dynamics of the system.

We see at chapter 2 that regressors selection, however, is not a simple task. If the nonlinear degree, the order of the model and the number inputs increases, the number of candidate models becomes too large for brute force approach. Considering the MIMO case, this problem is far worse than the SISO one if many inputs and outputs are required. The number of all different models can be calculated as

$$
\begin{align}
    n_m =
    \begin{cases}
    2^{n_r} & \text{for SISO models}, \\
    2^{n_{{_{m}}r}} & \text{for MIMO models},
    \end{cases}
\end{align}
\tag{5}
$$

where $n_r$ and  and $n_{{_{m}}r}$ are the values computed using the equations presented in Chapter 2.

A classical solution to regressors selection problem is the Forward Regression Orthogonal Least Squares (FROLS) algorithm associated with Error Reduction Ratio (ERR) algorithm. This technique is based on the Prediction Error Minimization framework and, one at time, select the most relevant regressor by using a step-wise regression. The FROLS method adapt the set of regressors in the search space into a set of orthogonal vectors, which ERR evaluates the individual contribution to the desired output variance.

## The Forward Regression Orthogonal Least Squares Algorithm

Consider the general NARMAX model defined in Equation 2.23 described in a generic form as

$$
\begin{equation}
    y_k = \psi^\top_{k-1}\hat{\Theta} + \xi_k,
\end{equation}
\tag{6}
$$

where $\psi^\top_{k-1} \in \mathbb{R}^{n_r \times n}$ is a vector of some combinations of the regressors and $\hat{\Theta} \in \mathbb{R}^{n_{\Theta}}$ the vector of estimated parameters. In a more compact form, the NARMAX model can be represented in a matrix form as:

$$
\begin{equation}
    y = \Psi\hat{\Theta} + \Xi,
\end{equation}
\tag{7}
$$

where

$$
\begin{align}
    Y = \begin{bmatrix}
    y_1 \\
    y_2 \\
    \vdots \\
    y_n
    \end{bmatrix},
    \Psi = \begin{bmatrix}
    \psi_{{_1}} \\
    \psi_{{_2}} \\
    \vdots \\
    \psi_{{_{n_{\Theta}}}}
    \end{bmatrix}^\top=
    \begin{bmatrix}
    \psi_{{_1}1} & \psi_{{_2}1} & \dots & \psi_{{_{n_{\Theta}}}1} \\
    \psi_{{_1}2} & \psi_{{_2}2} & \dots & \psi_{{_{n_{\Theta}}}2} \\
    \vdots & \vdots &       & \vdots \\
    \psi_{{_1}n} & \psi_{{_2}n} & \dots & \psi_{{_{n_{\Theta}}}n} \\
    \end{bmatrix},
    \hat{\Theta} = \begin{bmatrix}
    \hat{\Theta}_1 \\
    \hat{\Theta}_2 \\
    \vdots \\
    \hat{\Theta}_{n_\Theta}
    \end{bmatrix},
    \Xi = \begin{bmatrix}
    \xi_1 \\
    \xi_2 \\
    \vdots \\
    \xi_n
    \end{bmatrix}.
\end{align}
\tag{8}
$$

The parameters in equation above could be estimated as a result of a Least Squares-based algorithm, but this would require to optimize all parameters at the same time on account of the fact of interaction between regressors due to non-orthogonality characteristic. Consequently, the computational demand becomes impractical for high number of regressors. In this respect, the FROLS transforms the non-orthogonal model presented in the equation above into a orthogonal one.

The regressor matrix $\Psi$ can be orthogonally decomposed as

$$
\begin{equation}
    \Psi = QA,
\end{equation}
\tag{9}
$$

where $A \in \mathbb{R}^{n_{\Theta}\times n_{\Theta}}$ is an unit upper triangular matrix according to

$$
\begin{align}
A =
    \begin{bmatrix}
    1       & a_{12} & a_{13} & \dotsc & a_{1n_{\Theta}} \\
    0       &   1    & a_{23} & \dotsc & a_{2n_{\Theta}} \\
    0       &   0    &   1    & \dotsc &     \vdots       \\
    \vdots  & \vdots & \vdots & \ddots & a_{n_{\Theta}-1n_{\Theta}} \\
    0       &  0     &  0     &  0     & 1
    \end{bmatrix},
\end{align}
\tag{10}
$$

and $Q \in \mathbb{R}^{n\times n_{\Theta}}$ is a matrix with orthogonal columns $q_i$, described as

$$
\begin{equation}
    Q =
        \begin{bmatrix}
        q_{{_1}} & q_{{_2}} & q_{{_3}} & \dotsc & q_{{_{n_{\Theta}}}}
        \end{bmatrix},
\end{equation}
\tag{11}
$$

such that $Q^\top Q = \Lambda$ and $\Lambda$ is diagonal with entry $d_i$ and can be expressed as:

$$
\begin{align}
    d_i = q_i^\top q_i = \sum^{k=1}_{n}q_{{_i}k}q_{{_i}k}, \qquad 1\leq i \leq n_{\Theta}.
\end{align}
$$

Because the space spanned by the orthogonal basis $Q$ (Equation 11) is the same as that spanned by the basis set $\Psi$ (Equation 8) (i.e, contains every linear combination of elements of such subspace), we can define the Equation 7 as

$$
\begin{equation}
    Y = \underbrace{(\Psi A^{-1})}_{Q}\underbrace{(A\Theta)}_{g}+ \Xi = Qg+\Xi,
\end{equation}
\tag{12}
$$

where $g\in \mathbb{R}^{n_\Theta}$ is an auxiliary parameter vector. The solution of the model described in Equation 12 is given by

$$
\begin{equation}
    g = \left(Q^\top Q\right)^{-1}Q^\top Y = \Lambda^{-1}Q^\top Y
\end{equation}
\tag{13}
$$

or

$$
\begin{equation}
    g_{{_i}} = \frac{q_{{_i}}^\top Y}{q_{{_i}}^\top q_{{_i}}}.
\end{equation}
\tag{14}
$$

Since the parameter $\Theta$ and $g$ satisfies the triangular system $A\Theta = g$, any orthogonalization method like Householder, Gram-Schmidt, modified Gram-Schmidt or Givens transformations can be used to solve the equation and estimate the original parameters. Assuming that $E[\Psi^\top \Xi] = 0$, the output variance can be derived by multiplying Equation 12 with itself and dividing by $n$, resulting in

$$
\begin{equation}
    \frac{1}{n}Y^\top Y = \underbrace{\frac{1}{n}\sum^{i = 1}_{n_{\Theta}}g_{{_i}}^2q^\top_{{_i}}q_{{_i}}}_{\text{{output explained by the regressors}}} + \underbrace{\frac{1}{n}\Xi^\top \Xi}_{\text{{unexplained variance}}}.
\end{equation}
\tag{15}
$$

Thus, the ERR due to the inclusion of the regressor $q_{{_i}}$ is expressed as:

$$
[\text{ERR}]_i = \frac{g_{i}^2 \cdot q_{i}^\top q_{i}}{Y^\top Y}, \qquad \text{for } i=1,2,\dotsc, n_\Theta.
$$


There are many ways to terminate the algorithm. An approach often used is stop the algorithm if the model output variance drops below some predetermined limit $\varepsilon$:

$$
\begin{equation}
    1 - \sum_{i = 1}^{n_{\Theta}}\text{ERR}_i \leq \varepsilon,
\end{equation}
\tag{17}
$$

### Keep it simple

For the sake of simplicity, let's present the FROLS along with simple examples to make the intuition clear. First, let define the ERR calculation and then explain the idea of the FRLOS in simple terms.

#### Orthogonal case

Consider the case where we have a set of inputs defined as $x_1, x_2, \ldots, x_n$ and an output called $y$. These inputs are orthogonal vectors.

Lets suppose that we want to create a model to approximate $y$  using $x_1, x_2, \ldots, x_n$, as follows:

$$
y=\hat{\theta}_1 x_1+\hat{\theta}_2 x_2+\ldots+\hat{\theta}_n x_n+e
\tag{18}
$$

where $\hat{\theta}_1, \hat{\theta}_2, \ldots, \hat{\theta}_n$ are parameters and $e$ is white noise and independent of $x$ and $y$ (remember the  $E[\Psi^\top \Xi] = 0$, in previous section). In this case, we can rewrite the equation above as

$$
y = \hat{\theta} x
\tag{19}
$$

so

$$
\left\langle x, y\right\rangle = \left\langle \hat{\theta} x, x\right\rangle = \hat{\theta} \left\langle x, x\right\rangle
\tag{20}
$$

Which implies that

$$
\hat{\theta} = \frac{\left\langle x, y\right\rangle}{\left\langle x, x\right\rangle}
\tag{21}
$$

Therefore we can show that

$$
\begin{align}
& \left\langle x_1, y\right\rangle=\hat{\theta}_1\left\langle x_1, x_1\right\rangle \Rightarrow \hat{\theta}_1=\frac{\left\langle x_1, y\right\rangle}{\left\langle x_1, x_1\right\rangle}=\frac{x_1^T y}{x_1^T x_1} \\
& \left\langle x_2, y\right\rangle=\hat{\theta}_2\left\langle x_2, x_2\right\rangle \Rightarrow \hat{\theta}_2=\frac{\left\langle x_2, y\right\rangle}{\left\langle x_2, x_2\right\rangle}=\frac{x_2^T y}{x_2^T x_2}, \ldots \\
& \left\langle x_n, y\right\rangle=\hat{\theta}_n\left\langle x_n, x_n\right\rangle \Rightarrow \hat{\theta}_n=\frac{\left\langle x_n, y\right\rangle}{\left\langle x_n, x_n\right\rangle}=\frac{x_n^T y}{x_n^T x_n},
\end{align}
\tag{22}
$$


Following the same idea, we can also show that

$$
\langle y, y\rangle=\hat{\theta}_1^2\left\langle x_1, x_1\right\rangle+\hat{\theta}_2^2\left\langle x_2, x_2\right\rangle+\ldots+\hat{\theta}_n^2\left\langle x_n, x_n\right\rangle+\langle e, e\rangle
\tag{23}
$$

which can be described as

$$
y^T y=\hat{\theta}_1^2 x_1^T x_1+\hat{\theta}_2^2 x_2^T x_2+\ldots+\hat{\theta}_n^2 x_n^T x_n+e^T e
\tag{24}
$$

or

$$
\|y\|^2=\hat{\theta}_1^2\left\|x_1\right\|^2+\hat{\theta}_2^2\left\|x_2\right\|^2+\ldots+\hat{\theta}_n^2\left\|x_n\right\|^2+\|e\|^2
\tag{25}
$$

So, dividing both sides of the equation by $y$ and rearranging the equation, we have

$$
\frac{\|e\|^2}{\|y\|^2}=1-\hat{\theta}_1^2 \frac{\left\|x_1\right\|^2}{\|y\|^2}-\hat{\theta}_2^2 \frac{\left\|x_2\right\|^2}{\|y\|^2}-\ldots-\hat{\theta}_n^2 \frac{\left\|x_n\right\|^2}{\|y\|^2}
\tag{26}
$$

Because $\hat{\theta}_k=\frac{x_k^T y}{x_k^T x_k}=\frac{x_k^T y}{\left\|x_k\right\|^2}, k=1,2, . ., n$, we have

$$
\begin{align}
\frac{\|e\|^2}{\|y\|^2} & =1-\left(\frac{x_1^T y}{\left\|x_1\right\|^2}\right)^2 \frac{\left\|x_1\right\|^2}{\|y\|^2}-\left(\frac{x_2^T y}{\left\|x_2\right\|^2}\right)^2 \frac{\left\|x_2\right\|^2}{\|y\|^2}-\ldots-\left(\frac{x_n^T y}{\left\|x_n\right\|^2}\right)^2 \frac{\left\|x_n\right\|^2}{\|y\|^2} \\
& =1-\frac{\left(x_1^T y\right)^2}{\left\|x_1\right\|\left\|^2\right\| y \|^2}-\frac{\left(x_2^T y\right)^2}{\left\|x_2\right\|^2\|y\|^2}-\cdots-\frac{\left(x_n^T y\right)^2}{\left\|x_n\right\|^2\|y\|^2} \\
& =1-E R R_1 \quad-E R R_2-\cdots-E R R_n
\end{align}
\tag{27}
$$

where $\operatorname{ERR}_k(k=1,2 \ldots, n)$ is the Error Reduction Ratio defined in previous section.

Check the example bellow using the fundamental basis

```python
import numpy as np

y = np.array([3, 7, 8])
# Orthogonal Basis
x1 = np.array([1, 0, 0])
x2 = np.array([0, 1, 0])
x3 = np.array([0, 0, 1])

theta1 = (x1.T@y)/(x1.T@x1)
theta2 = (x2.T@y)/(x2.T@x2)
theta3 = (x3.T@y)/(x3.T@x3)

squared_y = y.T @ y
err1 = (x1.T@y)**2/((x1.T@x1) * squared_y)
err2 = (x2.T@y)**2/((x2.T@x2) * squared_y)
err3 = (x3.T@y)**2/((x3.T@x3) * squared_y)

print(f"x1 represents {round(err1*100, 2)}% of the variation in y, \n x2 represents {round(err2*100, 2)}% of the variation in y, \n x3 represents {round(err3*100, 2)}% of the variation in y")

x1 represents 7.38% of the variation in y,
x2 represents 40.16% of the variation in y,
x3 represents 52.46% of the variation in y
```

Lets see what happens in a non-orthogonal scenario.

```python
y = np.array([3, 7, 8])
x1 = np.array([1, 2, 2])
x2 = np.array([-1, 0, 2])
x3 = np.array([0, 0, 1])

theta1 = (x1.T@y)/(x1.T@x1)
theta2 = (x2.T@y)/(x2.T@x2)
theta3 = (x3.T@y)/(x3.T@x3)

squared_y = y.T @ y
err1 = (x1.T@y)**2/((x1.T@x1) * squared_y)
err2 = (x2.T@y)/((x2.T@x2) * squared_y)
err3 = (x3.T@y)**2/((x3.T@x3) * squared_y)

print(f"x1 represents {round(err1*100, 2)}% of the variation in y, \n x2 represents {round(err2*100, 2)}% of the variation in y, \n x3 represents {round(err3*100, 2)}% of the variation in y")

>>> x1 represents 99.18% of the variation in y,
>>> x2 represents 2.13% of the variation in y,
>>> x3 represents 52.46% of the variation in y
```
In this case, $x1$ have the highest $err$ value, so we have choose it to be the first orthogonal vector.

```python
q1 = x1.copy()

v1 = x2 - (q1.T@x2)/(q1.T@q1)*q1
errv1 = (v1.T@y)**2/((v1.T@v1) * squared_y)

v2 = x3 - (q1.T@x3)/(q1.T@q1)*q1
errv2 = (v2.T@y)**2/((v2.T@v2) * squared_y)

print(f"v1 represents {round(errv1*100, 2)}% of the variation in y, \n v2 represents {round(errv2*100, 2)}% of the variation in y")

>>> v1 represents 0.82% of the variation in y,
>>> v2 represents 0.66% of the variation in y
```

So, in this case, when we sum the err values of the first two orthogonal vectors, $x1$ and $v1$, we get $err_3 + errv1 = 100\%$. Then there is no need to keep the iterations looking for more terms. The model with this two terms already explain all the variance in the data.

> That's the idea of the FROLS algorithm. We calculate the ERR, choose the vector with the highest ERR to be the first orthogonal vector, orthogonalize every vector but the one we choose in the first step, calculate the ERR for each one of them, choose the vector with the highest ERR value and keep doing that until we reach some criteria.

In SysIdentPy, we have 2 hyperparameters called `n_terms` and `err_tol`. Both of them can be used to stop the iterations. The first one will iterate until `n_terms` are chosen. The second one iterate until the $\sum ERR_i > err_{tol}$ . If you set both, the algorithm stop when any of the conditions is true.

```python
model = FROLS(
        n_terms=50,
        ylag=7,
        xlag=7,
        basis_function=basis_function,
        err_tol=0.98
    )
```

SysIdentPy apply the Golub -Householder method for the orthogonal decomposition. A more detailed discussion about Householder and orthogonalization procedures in general can be found in [Chen, S. and Billings, S. A. and Luo, W.](https://www.tandfonline.com/doi/abs/10.1080/00207178908953472)

## Case Study

An example using real data will be described using SysIdentPy. In this example, we will build models linear and nonlinear models to describe the behavior of a DC motor operating as generator. Details of the experiment used to generate this data can be found in the paper (in Portuguese) [IDENTIFICAÇÃO DE UM MOTOR/GERADOR CC POR MEIO DE MODELOS POLINOMIAIS AUTORREGRESSIVOS E REDES NEURAIS ARTIFICIAIS](https://www.researchgate.net/publication/320418710_Identificacao_de_um_motorgerador_CC_por_meio_de_modelos_polinomiais_autorregressivos_e_redes_neurais_artificiais)

```python
import numpy as np
import pandas as pd
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function._basis_function import Polynomial
from sysidentpy.parameter_estimation import LeastSquares
from sysidentpy.utils.display_results import results
from sysidentpy.utils.plotting import plot_results

df1 = pd.read_csv("examples/datasets/x_cc.csv")
df2 = pd.read_csv("examples/datasets/y_cc.csv")

# checking the ouput
df2[5000:80000].plot(figsize=(10, 4))
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/generator_example.png?raw=true)
> Figure 2. Output of the electromechanical system.


In this example, we will decimate the data using $d = 500$. The rationale behind decimation here is that the data is oversampled due to the experimental setup. A future section will provide a detailed explanation of how to handle oversampled data in the context of system identification. For now, consider this approach as the most appropriate solution.

```python
x_train, x_valid = np.split(df1.iloc[::500].values, 2)
y_train, y_valid = np.split(df2.iloc[::500].values, 2)
```

In this case, we will build a NARX model. In SysIdentPy, this means setting `unbiased=False` in the `LeastSquares` definition. We'll use a `Polynomial` basis function and set the maximum lag for both input and output to 2. This configuration results in 15 terms in the information matrix, so we'll set `n_terms=15`. This specification is necessary because, in this example, `order_selection` is set to `False`. We will discuss `order_selection` in more detail in the Information Criteria section later on.

> `order_selection` is `True` by default in SysIdentPy. When `order_selection=False` the user must pass a values to `n_terms` because it is an optional argument and its default value is `None`. If we set `n_terms=5`, for exemple, the FROLS will stop after choosing the first 5 regressors. We do not want that in this case because we want the FROLS stop only when `e_tol` is reached.

```python
basis_function = Polynomial(degree=2)

model = FROLS(
	order_selection=False,
    ylag=2,
    xlag=2,
    estimator=LeastSquares(unbiased=False),
    basis_function=basis_function,
    e_tol=0.9999
    n_terms=15
)
```

SysIdentPy aims to simplify the use of algorithms like `FROLS` for the user. Building, training, or fitting a model is made straightforward through a simple interface called `fit`. By using this method, the entire process is handled internally, requiring no further interaction from the user.

```python
model.fit(X=x_train, y=y_train)
```

SysIdentPy also offers a method to retrieve detailed information about the fitted model. Users can check the terms included in the model, the estimated parameters, the Error Reduction Ratio (ERR) values, and more.

> We're using `pandas` here only to make the output more readable, but it's optional.

```python
r = pd.DataFrame(
    results(
        model.final_model,
        model.theta,
        model.err,
        model.n_terms,
        err_precision=8,
        dtype="sci",
    ),
    columns=["Regressors", "Parameters", "ERR"],
)

print(r)
```

| Regressors     | Parameters  | ERR            |
| -------------- | ----------- | -------------- |
| y(k-1)         | 1.0998E+00  | 9.86000384E-01 |
| x1(k-1)^2      | 1.0165E+02  | 7.94805130E-03 |
| y(k-2)^2       | -1.9786E-05 | 2.50905908E-03 |
| x1(k-1)y(k-1)  | -1.2138E-01 | 1.43301039E-03 |
| y(k-2)         | -3.2621E-01 | 1.02781443E-03 |
| x1(k-1)y(k-2)  | 5.3596E-02  | 5.35200312E-04 |
| x1(k-2)        | 3.4655E+02  | 2.79648078E-04 |
| x1(k-2)y(k-1)  | -5.1647E-02 | 1.12211942E-04 |
| x1(k-2)x1(k-1) | -8.2162E+00 | 4.54743448E-05 |
| y(k-2)y(k-1)   | 4.0961E-05  | 3.25346101E-05 |
>Table 1

The table above shows that 10 regressors (out of the 15 available) were needed to reach the defined `e_tol`, with the sum of the ERR for the selected regressors being $0.99992$.

Next, let's evaluate the model's performance using the test data. Similar to the `fit` method, SysIdentPy provides a `predict` method. To obtain the predicted values and plot the results, simply follow these steps:

```python
yhat = model.predict(X=x_valid, y=y_valid)
# plot only the first 100 samples (n=100)
plot_results(y=y_valid, yhat=yhat, n=100)
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/generator_predict_c4.png?raw=true)
> Figure 3. Free run simulation (or infinity-steps ahead prediction) of the fitted model.

## Information Criteria

We said that there are many ways to terminate the algorithm and select the model terms, but only ERR criteria was defined in previous section. Different ways to terminate the algorithm is by using some information criteria, e.g, Akaike Information Criteria (AIC). For Least Squares based regression analysis, the AIC indicates the number of regressors by minimizing the objective function ([Akaike, H.]([A new look at the statistical model identification | IEEE Journals & Magazine | IEEE Xplore](https://ieeexplore.ieee.org/document/1100705))):

$$
\begin{equation}
    J_{\text{AIC}} = \underbrace{n\log\left(Var[\xi_k]\right)}_{\text{first component}}+\underbrace{2n_{\Theta}}_{\text{{second component}}}.
\end{equation}
\tag{28}
$$

It is important to note that the equation above illustrates a trade-off between model fit and model complexity. Specifically, this trade-off involves balancing the model's ability to accurately fit the data (the first component) against its complexity, which is related to the number of parameters included (the second component). As additional terms are included in the model, the Akaike Information Criterion (AIC) value initially decreases, reaching a minimum that represents an optimal balance between model complexity and predictive accuracy. However, if the number of parameters becomes excessive, the penalty for complexity outweighs the benefit of a better fit, causing the AIC value to increase. The AIC and many others variants have been extensively used for linear and nonlinear system identification. Check [Wei, H. and Zhu, D. and Billings, S. A. and Balikhin, M. A.]([Forecasting the geomagnetic activity of the Dst index using multiscale radial basis function networks](https://www.sciencedirect.com/science/article/abs/pii/S0273117707002086)), [Martins, S. A. M. and Nepomuceno, E. G. and Barroso, M. F. S.]([Improved Structure Detection For Polynomial NARX Models Using a Multiobjective Error Reduction Ratio](https://link.springer.com/article/10.1007/s40313-013-0071-9)), [Hafiz, F. and Swain, A. and Mendes, E. M. A. M. and Patel, N.]([Structure Selection of Polynomial NARX Models Using Two Dimensional (2D) Particle Swarms](https://ieeexplore.ieee.org/document/8477782)), [Gu, Y. and Wei, H. and Balikhin, M. M.]([Nonlinear predictive model selection and model averaging using information criteria](https://www.tandfonline.com/doi/full/10.1080/21642583.2018.1496042)) and references therein.

Despite their effectiveness in many linear model selection scenarios, information criteria such as AIC can struggle to select an appropriate number of parameters when dealing with systems exhibiting significant nonlinear behavior. Additionally, these criteria may lead to suboptimal models if the search space does not encompass all the necessary terms required to accurately represent the _true_ model. Consequently, in highly nonlinear systems or when critical model components are missing, information criteria might not provide reliable guidance, resulting in models that exhibit poor performance.

Besides AIC, SysIdentPy provides other four different information criteria: [Bayesian Information Criteria](https://wires.onlinelibrary.wiley.com/doi/abs/10.1002/wics.199) (BIC), [Final Prediction Error](https://www.researchgate.net/publication/303679484_Introducao_a_Identificacao_de_Sistemas) (FPE), [Low of Iterated Logarithm Criteria](https://www.sciencedirect.com/science/article/abs/pii/S0169743902000515) (LILC), and [Corrected Akaike Information Criteria](https://www.sciencedirect.com/science/article/abs/pii/S0167715296001289) (AICc), which can be described respectively as

$$
\begin{align}
\operatorname{FPE}\left(n_\theta\right) & =N \ln \left[\sigma_{\text {erro }}^2\left(n_\theta\right)\right]+N \ln \left[\frac{N+n_\theta}{N-n_\theta}\right] \\
B I C\left(n_\theta\right) & =N \ln \left[\sigma_{\text {erro }}^2\left(n_\theta\right)\right]+n_\theta \ln N \\
A I C c &=A I C+2 n_p * \frac{n_p+1}{N-n_p-1} \\
LILC &= 2n_{\theta}\ln(\ln(N)) + N \ln(\left[\sigma_{\text {erro }}^2\left(n_\theta\right)\right])
\end{align}
\tag{29}
$$

To use any information criteria in SysIdentPy, set `order_selection=True` (as said before, the default value is already `True`). Besides `order_selection`, you can define how many regressors you want to evaluate before stopping the algorithm by using the `n_info_values` hyperparameter. The default value is $15$, but the user should increase it based on how many regressors exists given the `ylag`, `xlag` and the degree of the basis function.

> Using information Criteria can take a long time depending on how many regressors you are evaluating and the number of samples. To calculate the criteria, the ERR algorithm is executed `n` times where `n` is the number defined in `n_info_values`. Make sure to understand how it works to define whether you have to use it or not.

Running the same example, but now using the BIC information criteria to select the order of the model, we have

```python
model = FROLS(
    order_selection=True,
    n_info_values=15,
    ylag=2,
    xlag=2,
    info_criteria="bic",
    estimator=LeastSquares(unbiased=False),
    basis_function=basis_function
)
model.fit(X=x_train, y=y_train)
r = pd.DataFrame(
    results(
        model.final_model,
        model.theta,
        model.err,
        model.n_terms,
        err_precision=8,
        dtype="sci",
    ),
    columns=["Regressors", "Parameters", "ERR"],
)

print(r)
```

| Regressors       | Parameters  | ERR           |
|------------------|-------------|---------------|
| y(k-1)           | 1.3666E+00  | 9.86000384E-01|
| x1(k-1)^2        | 1.0500E+02  | 7.94805130E-03|
| y(k-2)^2         | -5.8577E-05 | 2.50905908E-03|
| x1(k-1)y(k-1)    | -1.2427E-01 | 1.43301039E-03|
| y(k-2)           | -5.1414E-01 | 1.02781443E-03|
| x1(k-1)y(k-2)    | 5.3001E-02  | 5.35200312E-04|
| x1(k-2)          | 3.1144E+02  | 2.79648078E-04|
| x1(k-2)y(k-1)    | -4.8013E-02 | 1.12211942E-04|
| x1(k-2)x1(k-1)   | -8.0561E+00 | 4.54743448E-05|
| x1(k-2)y(k-2)    | 4.1381E-03  | 3.25346101E-05|
| 1                | -5.6653E+01 | 7.54107553E-06|
| y(k-2)y(k-1)     | 1.5679E-04  | 3.52002717E-06|
| y(k-1)^2         | -9.0164E-05 | 6.17373260E-06|
>Table 2

In this case, instead of 8 regressors, the final model have 13 terms.

Currently, the number of regressors is determined by identifying the index of the last value where the difference between the current and previous value is less than 0. To inspect these values, you can use the following approach:

```python
xaxis = np.arange(1, model.n_info_values + 1)
plt.plot(xaxis, model.info_values)
plt.xlabel("n_terms")
plt.ylabel("Information Criteria")
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/dc_generator_aic_c4.png?raw=true)
> Figure 4. The plot shows the Information Criterion values (BIC) as a function of the number of terms included in the model. The model selection process, using the BIC criterion, iteratively adds regressors until the BIC reaches a minimum, indicating the optimal balance between model complexity and fit. The point where the BIC value stops decreasing marks the optimal number of terms, resulting in a final model with 13 terms.

The model prediction in this case is shown in Figure 5

```python
yhat = model.predict(X=x_valid, y=y_valid)
# plot only the first 100 samples (n=100)
plot_results(y=y_valid, yhat=yhat, n=100)
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/dc_generator_bic_c4.png?raw=true)
> Figure 5. Free run simulation (or infinity-steps ahead prediction) of the fitted model using BIC.

### Overview of the Information Criteria Methods

In this section, simulated data are used to provide users with a clearer understanding of the information criteria available in SysIdentPy.

> Here, we're working with a known model structure, which allows us to focus on how different information criteria perform. When dealing with real data, the correct number of terms in the model is unknown, making these methods invaluable for guiding model selection.

> If you review the metrics below, you'll notice excellent performance across all models. However, it's crucial to remember that System Identification is about finding the optimal model structure. Model Structure Selection is at the heart of NARMAX methods!

The data is generated by simulating the following model:

$$
y_k = 0.2y_{k-1} + 0.1y_{k-1}x_{k-1} + 0.9x_{k-1} + e_k
\tag{30}
$$

If `colored_noise` is set to `True`, the noise term is defined as:

$$
e_k = 0.8\nu_{k-1} + \nu_k
\tag{31}
$$

where $x$ is a uniformly distributed random variable and $\nu$ is a Gaussian-distributed variable with $\mu = 0$ and $\sigma = 0.1$.

In the next example, we will generate data with 100 samples, using white noise, and select 70% of the data to train the model.

```python
import numpy as np
import pandas as pd
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function._basis_function import Polynomial
from sysidentpy.utils.generate_data import get_siso_data
from sysidentpy.utils.display_results import results


x_train, x_valid, y_train, y_valid = get_siso_data(
    n=100, colored_noise=False, sigma=0.1, train_percentage=70
)
```

The idea is to show the impact of the information criteria to select the number of terms to compose the final model. You will se why it is an auxiliary tool and let the algorithm select the number of terms based on the minimum value is not always a good idea when dealing with data highly corrupted by noise (even white noise).

#### AIC

```python
basis_function = Polynomial(degree=2)
model = FROLS(
    order_selection=True,
    ylag=2,
    xlag=2,
    info_criteria="aic",
    basis_function=basis_function,
)

model.fit(X=x_train, y=y_train)
yhat = model.predict(X=x_valid, y=y_valid)

r = pd.DataFrame(
    results(
        model.final_model,
        model.theta,
        model.err,
        model.n_terms,
        err_precision=8,
        dtype="sci",
    ),
    columns=["Regressors", "Parameters", "ERR"],
)

print(r)

xaxis = np.arange(1, model.n_info_values + 1)
plt.plot(xaxis, model.info_values)
plt.xlabel("n_terms")
plt.ylabel("Information Criteria")
```

The regressors, the free run simulation and the AIC values are detailed bellow.

| Regressors     | Parameters  | ERR            |
| -------------- | ----------- | -------------- |
| x1(k-2)        | 9.4236E-01  | 9.26094341E-01 |
| y(k-1)         | 2.4933E-01  | 3.35898283E-02 |
| x1(k-1)y(k-1)  | 1.3001E-01  | 2.35736200E-03 |
| x1(k-1)        | 8.4024E-02  | 4.11741791E-03 |
| x1(k-1)^2      | 7.0807E-02  | 2.54231877E-03 |
| x1(k-2)^2      | -9.1138E-02 | 1.39658893E-03 |
| y(k-1)^2       | 1.1698E-01  | 1.70257419E-03 |
| x1(k-2)y(k-2)  | 8.3745E-02  | 1.11056684E-03 |
| y(k-2)^2       | -4.1946E-02 | 1.01686239E-03 |
| x1(k-2)x1(k-1) | 5.9034E-02  | 7.47435512E-04 |
>Table 3

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/predict_aic_c4.png?raw=true)
> Figure 5. Free run simulation (or infinity-steps ahead prediction) of the fitted model using AIC.

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/aic_c4.png?raw=true)
> Figure 6. The plot shows the Information Criterion values (AIC) as a function of the number of terms included in the model. The model selection process, using the AIC criterion, iteratively adds regressors until the AIC reaches a minimum, indicating the optimal balance between model complexity and fit. The point where the AICc value stops decreasing marks the optimal number of terms, resulting in a final model with 10 terms.

For this case, we have a model with 10 terms. We know that the correct number is 3 because of the simulated system we are using as example.

#### AICc

The only change we have to do to use AICc instead of AIC is changing the information criteria hyperparameter: `information_criteria="aicc"`

```python
basis_function = Polynomial(degree=2)
model = FROLS(
    order_selection=True,
    n_info_values=15,
    ylag=2,
    xlag=2,
    info_criteria="aicc",
    basis_function=basis_function,
)
model.fit(X=x_train, y=y_train)
yhat = model.predict(X=x_valid, y=y_valid)
r = pd.DataFrame(
    results(
        model.final_model,
        model.theta,
        model.err,
        model.n_terms,
        err_precision=8,
        dtype="sci",
    ),
    columns=["Regressors", "Parameters", "ERR"],
)
print(r)
plot_results(y=y_valid, yhat=yhat, n=1000)

xaxis = np.arange(1, model.n_info_values + 1)
plt.plot(xaxis, model.info_values)
plt.xlabel("n_terms")
plt.ylabel("Information Criteria")
```

| Regressors       | Parameters  | ERR           |
|------------------|-------------|---------------|
| x1(k-2)          | 9.2282E-01  | 9.26094341E-01|
| y(k-1)           | 2.4294E-01  | 3.35898283E-02|
| x1(k-1)y(k-1)    | 1.2753E-01  | 2.35736200E-03|
| x1(k-1)          | 6.9597E-02  | 4.11741791E-03|
| x1(k-1)^2        | 7.0578E-02  | 2.54231877E-03|
| x1(k-2)^2        | -1.0523E-01 | 1.39658893E-03|
| y(k-1)^2         | 1.0949E-01  | 1.70257419E-03|
| x1(k-2)y(k-2)    | 7.1821E-02  | 1.11056684E-03|
| y(k-2)^2         | -3.9756E-02 | 1.01686239E-03|
>Table 4


![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/predicted_aicc_c4.png?raw=true)
> Figure 7. Free run simulation (or infinity-steps ahead prediction) of the fitted model using AICc.

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/aicc_c4.png?raw=true)
> Figure 8. The plot shows the Information Criterion values (AICc) as a function of the number of terms included in the model. The model selection process, using the AIC criterion, iteratively adds regressors until the AICc reaches a minimum, indicating the optimal balance between model complexity and fit. The point where the AICc value stops decreasing marks the optimal number of terms, resulting in a final model with 9 terms.

This time we have a model with 9 regressors.

#### BIC

```python
basis_function = Polynomial(degree=2)
model = FROLS(
    order_selection=True,
    n_info_values=15,
    ylag=2,
    xlag=2,
    info_criteria="bic",
    basis_function=basis_function,
)
model.fit(X=x_train, y=y_train)
yhat = model.predict(X=x_valid, y=y_valid)
r = pd.DataFrame(
    results(
        model.final_model,
        model.theta,
        model.err,
        model.n_terms,
        err_precision=8,
        dtype="sci",
    ),
    columns=["Regressors", "Parameters", "ERR"],
)
print(r)
plot_results(y=y_valid, yhat=yhat, n=1000)

xaxis = np.arange(1, model.n_info_values + 1)
plt.plot(xaxis, model.info_values)
plt.xlabel("n_terms")
plt.ylabel("Information Criteria")
```

| Regressors | Parameters | ERR            |
| ---------- | ---------- | -------------- |
| x1(k-2)    | 9.1726E-01 | 9.26094341E-01 |
| y(k-1)     | 1.8670E-01 | 3.35898283E-02 |
>Table 5


![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/predicted_bic_c4.png?raw=true)
> Figure 9. Free run simulation (or infinity-steps ahead prediction) of the fitted model using BIC.

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/bic_c4.png?raw=true)
> Figure 10. The plot shows the Information Criterion values (BIC) as a function of the number of terms included in the model. The model selection process, using the BIC criterion, iteratively adds regressors until the BIC reaches a minimum, indicating the optimal balance between model complexity and fit. The point where the BIC value stops decreasing marks the optimal number of terms, resulting in a final model with 2 terms.

BIC returned a model with only 2 regressors!

#### LILC

```python
basis_function = Polynomial(degree=2)
model = FROLS(
    order_selection=True,
    n_info_values=15,
    ylag=2,
    xlag=2,
    info_criteria="lilc",
    basis_function=basis_function,
)
model.fit(X=x_train, y=y_train)
yhat = model.predict(X=x_valid, y=y_valid)
r = pd.DataFrame(
    results(
        model.final_model,
        model.theta,
        model.err,
        model.n_terms,
        err_precision=8,
        dtype="sci",
    ),
    columns=["Regressors", "Parameters", "ERR"],
)
print(r)
plot_results(y=y_valid, yhat=yhat, n=1000)

xaxis = np.arange(1, model.n_info_values + 1)
plt.plot(xaxis, model.info_values)
plt.xlabel("n_terms")
plt.ylabel("Information Criteria")
```

| Regressors    | Parameters  | ERR            |
| ------------- | ----------- | -------------- |
| x1(k-2)       | 9.1160E-01  | 9.26094341E-01 |
| y(k-1)        | 2.3178E-01  | 3.35898283E-02 |
| x1(k-1)y(k-1) | 1.2080E-01  | 2.35736200E-03 |
| x1(k-1)       | 6.3113E-02  | 4.11741791E-03 |
| x1(k-1)^2     | 5.4088E-02  | 2.54231877E-03 |
| x1(k-2)^2     | -9.0683E-02 | 1.39658893E-03 |
| y(k-1)^2      | 8.2157E-02  | 1.70257419E-03 |
>Table 6

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/predicted_lilc_c4.png?raw=true)
> Figure 11. Free run simulation (or infinity-steps ahead prediction) of the fitted model using LILC.


![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/lilc_c4.png?raw=true)
>Figure 12. The plot shows the Information Criterion values (LILC) as a function of the number of terms included in the model. The model selection process, using the LILC criterion, iteratively adds regressors until the LILC reaches a minimum, indicating the optimal balance between model complexity and fit. The point where the LILC value stops decreasing marks the optimal number of terms, resulting in a final model with 7 terms.

LILC returned a model with 7 regressors.

#### FPE

```python
basis_function = Polynomial(degree=2)
model = FROLS(
    order_selection=True,
    n_info_values=15,
    ylag=2,
    xlag=2,
    info_criteria="fpe",
    basis_function=basis_function,
)
model.fit(X=x_train, y=y_train)
yhat = model.predict(X=x_valid, y=y_valid)
r = pd.DataFrame(
    results(
        model.final_model,
        model.theta,
        model.err,
        model.n_terms,
        err_precision=8,
        dtype="sci",
    ),
    columns=["Regressors", "Parameters", "ERR"],
)
print(r)
plot_results(y=y_valid, yhat=yhat, n=1000)

xaxis = np.arange(1, model.n_info_values + 1)
plt.plot(xaxis, model.info_values)
plt.xlabel("n_terms")
plt.ylabel("Information Criteria")
```

| Regressors     | Parameters  | ERR            |
| -------------- | ----------- | -------------- |
| x1(k-2)        | 9.4236E-01  | 9.26094341E-01 |
| y(k-1)         | 2.4933E-01  | 3.35898283E-02 |
| x1(k-1)y(k-1)  | 1.3001E-01  | 2.35736200E-03 |
| x1(k-1)        | 8.4024E-02  | 4.11741791E-03 |
| x1(k-1)^2      | 7.0807E-02  | 2.54231877E-03 |
| x1(k-2)^2      | -9.1138E-02 | 1.39658893E-03 |
| y(k-1)^2       | 1.1698E-01  | 1.70257419E-03 |
| x1(k-2)y(k-2)  | 8.3745E-02  | 1.11056684E-03 |
| y(k-2)^2       | -4.1946E-02 | 1.01686239E-03 |
| x1(k-2)x1(k-1) | 5.9034E-02  | 7.47435512E-04 |
>Table 7

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/predicted_fpe_c4.png?raw=true)
> Figure 13. Free run simulation (or infinity-steps ahead prediction) of the fitted model using FPE.


![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/fpe_c4.png?raw=true)
> Figure 14. The plot shows the Information Criterion values (FPE) as a function of the number of terms included in the model. The model selection process, using the FPE criterion, iteratively adds regressors until the FPE reaches a minimum, indicating the optimal balance between model complexity and fit. The point where the FPE value stops decreasing marks the optimal number of terms, resulting in a final model with 10 terms.

FPE returned a model with 10 regressors.

## Meta Model Structure Selection (MetaMSS)

> This section largely reflects content from a paper I published on [ArXiv](https://arxiv.org/abs/2109.09917) titled *"Meta-Model Structure Selection: Building Polynomial NARX Models for Regression and Classification."* This paper was initially written for journal publication based on the results of my [master's thesis](https://ufsj.edu.br/portal2-repositorio/File/ppgel/225-2020-02-17-DissertacaoWilsonLacerda.pdf). However, as I transitioned into a Data Scientist role and considering the lengthy journal submission process and academic delays, I decided not to pursue journal publication at this time. Thus, the paper remains available only on ArXiv.

> The work extends a previous paper I presented at a [Brazilian conference](https://proceedings.science/sbai-2019/trabalhos/identificacao-de-sistemas-nao-lineares-utilizando-o-algoritmo-hibrido-e-binario?lang=pt-br) (in Portuguese), where part of the results were initially shared.

This section introduces a meta-heuristic approach for selecting the structure of polynomial NARX models in regression tasks. The proposed method considers both the complexity of the model and the contribution of each term to construct parsimonious models through a novel cost function formulation. The robustness of this new algorithm is evaluated using various simulated and experimental systems with different nonlinear characteristics. The results demonstrate that the algorithm effectively identifies the correct model when the true structure is known and produces parsimonious models for experimental data, even in cases where traditional and contemporary methods often fail. The new approach is compared against classical methods such as FROLS and recent randomized techniques.

We mentioned that selecting the appropriate model terms is crucial for accurately capturing the dynamics of the original system. Challenges such as overparameterization and numerical ill-conditioning often arise due to the limitations of existing identification algorithms in selecting the right terms for the final model. Check [Aguirre, L. A. and Billings, S. A.]([Dynamical effects of overparametrization in nonlinear models](https://www.sciencedirect.com/science/article/abs/pii/0167278995900535)), [Piroddi, L. and Spinelli, W.]([An identification algorithm for polynomial NARX models based on simulation error minimization](https://www.tandfonline.com/doi/abs/10.1080/00207170310001635419)). We also mentioned that one of the most traditionally algorithms for structure selection of polynomial NARMAX is the ERR algorithm. Numerous variants of FROLS algorithm has been developed to improve the model selection performance such as [Billings, S. A. and Chen, S. and Korenberg, M. J.]([Identification of MIMO non-linear systems using a forward-regression orthogonal estimator](https://www.tandfonline.com/doi/abs/10.1080/00207178908559767)), [Farina, M. and Piroddi, L.]([Simulation Error Minimization–Based Identification of Polynomial Input–Output Recursive Models](https://www.sciencedirect.com/science/article/pii/S1474667016388462)), [Guo, Y. and Guo, L. Z. and Billings, S. A. and Wei, H.]([A New Iterative Orthogonal Forward Regression Algorithm](https://eprints.whiterose.ac.uk/107315/3/A%20New%20Iterative%20Orthogonal%20Forward%20Regression%20Algorithm%20-%20R2.pdf)), [Mao, K. Z. and Billings, S. A.]([VARIABLE SELECTION IN NON-LINEAR SYSTEMS MODELLING](https://www.sciencedirect.com/science/article/abs/pii/S0888327098901807)). The drawbacks of the FROLS have been extensively reviewed in the literature, e.g., in [Billings, S. A. and Aguirre, L. A.](https://core.ac.uk/download/pdf/29031334.pdf), [Palumbo, P. and Piroddi, L.](https://ui.adsabs.harvard.edu/abs/2001JSV...239..405P/abstract),  [Falsone, A. and Piroddi, L. and Prandini, M.](https://www.sciencedirect.com/science/article/abs/pii/S0005109815003088). Most of these weak points are related to (i) the Prediction Error Minimization (PEM) framework; (ii) the inadequacy of the ERR index in measuring the absolute importance of regressors; (iii) the use of information criteria such as AIC, FPE and the BIC, to select the model order. Regarding the information criteria, although these techniques work well for linear models, in a nonlinear context no simple relation between model size and accuracy can be established [Falsone, A. and Piroddi, L. and Prandini, M.]([A randomized algorithm for nonlinear model structure selection](https://www.sciencedirect.com/science/article/abs/pii/S0005109815003088)) , [Chen, S. and Hong, X. and Harris, C. J.]([Sparse kernel regression modeling using combined locally regularized orthogonal least squares and D-optimality experimental design](https://ieeexplore.ieee.org/document/1205199)).

Due to the limitations of Ordinary Least Squares (OLS)-based algorithms, recent research has presented solutions that diverged from the classical FROLS approach. New methods have reformulated the Model Structure Selection (MSS) process within a probabilistic framework and employed random sampling techniques [Falsone, A. and Piroddi, L. and Prandini, M.]([A randomized algorithm for nonlinear model structure selection](https://www.sciencedirect.com/science/article/abs/pii/S0005109815003088)), [Tempo, R. and Calafiore, G. and Dabbene, F.]([Randomized Algorithms for Analysis and Control of Uncertain Systems: With Applications](https://link.springer.com/book/10.1007/978-1-4471-4610-0)), [Baldacchino, T. and Anderson, S. R. and Kadirkamanathan, V.]([Computational system identification for Bayesian NARMAX modelling](https://www.sciencedirect.com/science/article/abs/pii/S0005109813003063)), [Rodriguez-Vazquez, K. and Fonseca, C. M. and Fleming, P. J.]([Identifying the structure of nonlinear dynamic systems using multiobjective genetic programming](https://ieeexplore.ieee.org/document/1306531)), [Severino, A. G. V. and Araujo, F. M. U. de](https://repositorio.ufrn.br/bitstream/123456789/24900/1/AlcemyGabrielVitorSeverino_DISSERT.pdf). Despite their advancements, these meta-heuristic and probabilistic approaches exhibit certain shortcomings. In particular, these methods often rely on information criteria such as AIC, FPE, and BIC to define the cost function for optimization, which frequently leads to over-parameterized models.

Consider $\mathcal{F}$ as a class of bounded functions $\phi: \mathbf{R} \mapsto \mathbf{R}$. If the properties of $\phi(x)$ satisfy

$$
\begin{align}
    &\lim\limits_{x \to \infty} \phi(x) = \alpha \nonumber \\
    &\lim\limits_{x \to -\infty} \phi(x) = \beta \quad \text{with } \alpha > \beta,  \nonumber
\end{align}
\tag{32}
$$

the function is called sigmoidal.

In this particular case and following definition Equation 32 with $alpha = 0$ and $\beta = 1$, we write a "S" shaped curve as

$$
\begin{equation}
    \varsigma(x) = \frac{1}{1+e^{-a(x-c)}}.
\end{equation}
\tag{33}
$$

In that case, we can specify $a$, the rate of change. If $a$ is close to zero, the sigmoid function will be gradual. If $a$ is large, the sigmoid function will have an abrupt or sharp transition. If $a$ is negative, the sigmoid will go from $1$ to zero. The parameter $c$ corresponds to the x value where $y = 0.5$.

The Sigmoid Linear Unit Function (SiLU) is defined by the sigmoid function multiplied by its input

$$
\begin{equation}
    \text{silu}(x) = x \varsigma(x),
\end{equation}
\tag{34}
$$

which can be viewed as an steeper sigmoid function with overshoot.

### Meta-heuristics

Over the past two decades, nature-inspired optimization algorithms have gained prominence due to their flexibility, simplicity, versatility, and ability to avoid local optima in real-world applications.

Meta-heuristic algorithms are characterized by two fundamental features: exploitation and exploration [Blum, C. and Roli, A.]([Metaheuristics in combinatorial optimization: Overview and conceptual comparison](https://dl.acm.org/doi/10.1145/937503.937505)). **Exploitation** focuses on utilizing local information to refine the search around the current best solution, improving the quality of nearby solutions. Conversely, **exploration** aims to search a broader area of the solution space to discover potentially superior solutions and prevent the algorithm from getting trapped in local optima.

Despite the lack of a universal consensus on the definitions of exploration and exploitation in evolutionary computing, as highlighted by [Eiben, Agoston E and Schippers, Cornelis A](https://www.researchgate.net/publication/220443407_On_Evolutionary_Exploration_and_Exploitation), it is generally agreed that these concepts function as opposing forces that are challenging to balance. To address this challenge, hybrid metaheuristics combine multiple algorithms to leverage both exploitation and exploration, resulting in more robust optimization methods.

#### The Binary hybrid Particle Swarm Optimization and Gravitational Search Algorithm (BPSOGSA) algorithm

Achieving a balance between exploration and exploitation is a significant challenge in most meta-heuristic algorithms. For this method, we enhance performance and flexibility in the search process by employing a hybrid approach that combines Binary Particle Swarm Optimization (BPSO) with Gravitational Search Algorithm (GSA), as proposed by [Mirjalili, S. and Hashim, S. Z. M.](https://ieeexplore.ieee.org/abstract/document/6141614). This hybrid method incorporates a low-level co-evolutionary heterogeneous technique originally introduced by [Talbi, E. G.](https://link.springer.com/article/10.1023/A:1016540724870).

The BPSOGSA approach leverages the strengths of both algorithms: the Particle Swarm Optimization (PSO) component is known to be good in exploring the entire search space to identify the global optimum, while the Gravitational Search Algorithm (GSA) component effectively refines the search by focusing on local solutions within a binary space. This combination aims to provide a more comprehensive and effective optimization strategy, ensuring a better balance between exploration and exploitation.

#### Standard Particle Swarm Optimization (PSO)

In Particle Swarm Optimization (PSO) [Kennedy, J. and Eberhart, R. C.](https://ieeexplore.ieee.org/document/488968), [Kennedy, J.](https://ieeexplore.ieee.org/document/488968), each particle represents a candidate solution and is characterized by two components: its position in the search space, denoted as $\vec{x}_{\,np,d} \in \mathbb{R}^{np \times d}$, and its velocity, $\vec{v}_{\,np,d} \in \mathbb{R}^{np \times d}$. Here, $np = 1, 2, \ldots, n_a$ where $n_a$ is the size of the swarm, and $d$ is the dimensionality of the problem. The initial population is represented as follows:

$$
\vec{x}_{\,np,d} =
\begin{bmatrix}
x_{1,1} & x_{1,2} & \cdots & x_{1,d} \\
x_{2,1} & x_{2,2} & \cdots & x_{2,d} \\
\vdots  & \vdots  & \ddots & \vdots \\
x_{n_a,1} & x_{n_a,2} & \cdots & x_{n_a,d}
\end{bmatrix}
\tag{35}
$$

At each iteration $t$, the position and velocity of a particle are updated using the following equations:

$$
v_{np,d}^{t+1} = \zeta v_{np,d}^{t} + c_1 \kappa_1 (pbest_{np}^{t} - x_{np,d}^{t})
+ c_2 \kappa_2 (gbest_{np}^{t} - x_{np,d}^{t}),
\tag{36}
$$

where $\kappa_j \in \mathbb{R}$ for $j = [1,2]$ are continuous random variables in the interval $[0,1]$, $\zeta \in \mathbb{R}$ is the inertia factor that controls the influence of the previous velocity on the current one and represents a trade-off between exploration and exploitation, $c_1$ is the cognitive factor associated with the personal best position $pbest$, and $c_2$ is the social factor associated with the global best position $gbest$. The velocity $\vec{v}_{\,np,d}$ is typically constrained within the range $[v_{min}, v_{max}]$ to prevent particles from moving outside the search space. The updated position is then computed as:

$$
x_{np,d}^{t+1} = x_{np,d}^{t} + v_{np,d}^{t+1}.
\tag{37}
$$

#### Standard Gravitational Search Algorithm (GSA)

In the Gravitational Search Algorithm (GSA) [Rashedi, Esmat and Nezamabadi-Pour, Hossein and Saryazdi, Saeid]([GSA: A Gravitational Search Algorithm](https://www.sciencedirect.com/science/article/abs/pii/S0020025509001200)), agents are represented by masses, where the magnitude of each mass is proportional to the fitness value of the agent. These masses interact through gravitational forces, attracting each other towards locations closer to the global optimum. Heavier masses (agents with better fitness) move more slowly, while lighter masses (agents with poorer fitness) move more rapidly. Each mass in GSA has four properties: position, inertial mass, active gravitational mass, and passive gravitational mass. The position of a mass represents a candidate solution to the problem, and its gravitational and inertial masses are derived from the fitness function.

Consider a population of agents as described by the following equations. At a specific time $t$, the velocity and position of each agent are updated as follows:

$$
\begin{align}
    v_{i,d}^{t+1} &= \kappa_i \times v_{i,d}^t + a_{i,d}^t, \\
    x_{i,d}^{t+1} &= x_{i,d}^t + v_{i,d}^{t+1}.
\end{align}
\tag{38}
$$

Here, $\kappa_i$ introduces stochastic characteristics to the search process. The acceleration $a_{i,d}^t$ is computed according to the law of motion [Rashedi, Esmat and Nezamabadi-Pour, Hossein and Saryazdi, Saeid](https://www.sciencedirect.com/science/article/abs/pii/S0020025509001200):

$$
\begin{equation}
    a_{i,d}^t = \frac{F_{i,d}^t}{M_{ii}^{t}},
\end{equation}
\tag{39}
$$

where $M_{ii}^{t}$ is the inertial mass of agent $i$ and $F_{i,d}^t$ represents the gravitational force acting on agent $i$ in the $d$-dimensional space. The detailed process for calculating and updating $F_{i,d}$ and $M_{ii}$ can be found in [Rashedi, Esmat and Nezamabadi-Pour, Hossein and Saryazdi, Saeid](https://www.sciencedirect.com/science/article/abs/pii/S0020025509001200).

#### The Binary Hybrid Optimization Algorithm

The combination of algorithms follows the approach described in [Mirjalili, S. and Hashim, S. Z. M.]([A new hybrid PSOGSA algorithm for function optimization](https://ieeexplore.ieee.org/abstract/document/6141614)):

$$
\begin{align}
    v_{i}^{t+1} = \zeta \times v_i^t + \mathrm{c}'_{1} \times \kappa \times a_i^t + \mathrm{c}'_2 \times \kappa \times (gbest - x_i^t),
\end{align}
\tag{40}
$$

where $\mathrm{c}'_j \in \mathbb{R}$ are acceleration coefficients. This formulation accelerates the exploitation phase by incorporating the best mass location found so far. However, this method may negatively impact the exploration phase. To address this issue, [Mirjalili, S. and Mirjalili, S. M. and Lewis, A.]([Grey Wolf Optimizer](https://www.sciencedirect.com/science/article/abs/pii/S0965997813001853)) proposed adaptive values for $\mathrm{c}'_j$, as described in [Mirjalili, S. and Wang, Gai-Ge and Coelho, L. dos S.]([Binary optimization using hybrid particle swarm optimization and gravitational search algorithm](https://dl.acm.org/doi/10.1007/s00521-014-1629-6)):

$$
\begin{align}
    \mathrm{c}_1' &= -2 \times \frac{t^3}{\max(t)^3} + 2, \\
    \mathrm{c}_2' &= 2 \times \frac{t^3}{\max(t)^3} + 2.
\end{align}
\tag{41}
$$

In each iteration, the positions of particles are updated according to the following rules, with continuous space mapped to discrete solutions using a transfer function ([Mirjalili, S. and Lewis, A.]([S-shaped versus V-shaped transfer functions for binary Particle Swarm Optimization](https://www.sciencedirect.com/science/article/abs/pii/S2210650212000648))):

$$
\begin{equation}
    S(v_{ik}) = \left|\frac{2}{\pi}\arctan\left(\frac{\pi}{2}v_{ik}\right)\right|.
\end{equation}
\tag{42}
$$

With a uniformly distributed random number $\kappa \in (0,1)$, the positions of the agents in the binary space are updated as follows:

$$
\begin{equation}
    x_{np,d}^{t+1} =
    \begin{cases}
        (x_{np,d}^{t})^{-1}, & \text{if } \kappa < S(v_{ik}^{t+1}), \\
        x_{np,d}^{t}, & \text{if } \kappa \geq S(v_{ik}^{t+1}).
    \end{cases}
\end{equation}
\tag{43}
$$

### Meta-Model Structure Selection (MetaMSS): Building NARX for Regression

In this section, we explore the meta-heuristic approach for selecting the structure of NARX models using BPSOGSA proposed in my [master's thesis](https://ufsj.edu.br/portal2-repositorio/File/ppgel/225-2020-02-17-DissertacaoWilsonLacerda.pdf). This method searches for the optimal model structure within a decision space defined by a predefined dictionary of regressors. The objective function for this optimization problem is based on the root mean squared error (RMSE) of the free-run simulation output, augmented by a penalty factor that accounts for the model's complexity and the contribution of each regressor to the final model.

#### Encoding Scheme

The process of using BPSOGSA for model structure selection involves defining the dimensions of the test function. Specifically, $n_y$, $n_x$, and $\ell$ are set to cover all possible regressors, and a general matrix of regressors, $\Psi$, is constructed. The number of columns in $\Psi$ is denoted as $noV$, and the number of agents, $N$, is specified. A binary $noV \times N$ matrix, referred to as $\mathcal{X}$, is then randomly generated to represent the position of each agent in the search space. Each column of $\mathcal{X}$ represents a potential solution, which is essentially a candidate model structure to be evaluated in each iteration. In this matrix, a value of 1 indicates that the corresponding column of $\Psi$ is included in the reduced matrix of regressors, while a value of 0 indicates exclusion.

For example, consider a case where all possible regressors are defined with $\ell = 1$ and $n_y = n_u = 2$. The matrix $\Psi$ is:

$$
\begin{align}
[ \text{constant} \quad y(k-1) \quad y(k-2) \quad u(k-1) \quad u(k-2) ]
\end{align}
\tag{44}
$$

With 5 possible regressors, $noV = 5$. Assuming $N = 5$, $\mathcal{X}$ might be represented as:

$$
\begin{equation}
    \mathcal{X} =
    \begin{bmatrix}
        0 & 1 & 0 & 0 & 0 \\
        1 & 1 & 1 & 0 & 1 \\
        0 & 0 & 1 & 1 & 0 \\
        0 & 1 & 0 & 0 & 1 \\
        1 & 0 & 1 & 1 & 0
    \end{bmatrix}
\end{equation}
\tag{45}
$$

Each column of $\mathcal{X}$ is transposed to generate a candidate solution. For example, the first column of $\mathcal{X}$ yields:

$$
\begin{equation*}
    \mathcal{X} =
    \begin{bmatrix}
        \text{constant} & y(k-1) & y(k-2) & u(k-1) & u(k-2) \\
        1 & 1 & 1 & 0 & 1
    \end{bmatrix}
\end{equation*}
\tag{46}
$$

In this scenario, the first model to be evaluated is $\alpha y(k-1) + \beta u(k-2)$, where $\alpha$ and $\beta$ are parameters estimated using some parameter estimation method. The process is repeated for each subsequent column of $\mathcal{X}$.

#### Formulation of the objective function

For each candidate model structure randomly defined, the linear-in-the-parameters system can be solved directly using the Least Squares algorithm or any other method available in SysIdentPy. The variance of the estimated parameters can be calculated as:

$$
\hat{\sigma}^2 = \hat{\sigma}_e^2 V_{jj},
\tag{47}
$$


where $\hat{\sigma}_e^2$ is the estimated noise variance, given by:

$$
\hat{\sigma}_e^2 = \frac{1}{N-m} \sum_{k=1}^{N} (y_k - \psi_{k-1}^\top \hat{\Theta}),
\tag{48}
$$


and $V_{jj}$ is the $j$th diagonal element of $(\Psi^\top \Psi)^{-1}$.

The estimated standard error of the $j$th regression coefficient $\hat{\Theta}_j$ is the positive square root of the diagonal elements of $\hat{\sigma}^2$:

$$
\mathrm{se}(\hat{\Theta}_j) = \sqrt{\hat{\sigma}^2_{jj}}.
\tag{49}
$$

To assess the statistical relevance of each regressor, a penalty test considers the standard error of the regression coefficients. In this case, the $t$-test is used to perform a hypothesis test on the coefficients, evaluating the significance of individual regressors in the multiple linear regression model. The hypothesis statements are:

$$
\begin{align*}
   H_0 &: \Theta_j = 0, \\
   H_a &: \Theta_j \neq 0.
\end{align*}
\tag{50}
$$

The $t$-statistic is computed as:

$$
T_0 = \frac{\hat{\Theta}_j}{\mathrm{se}(\hat{\Theta}_j)},
\tag{51}
$$

which measures how many standard deviations $\hat{\Theta}_j$ is from zero. More precisely, if:

$$
-t_{\alpha/2, N-m} < T_0 < t_{\alpha/2, N-m},
\tag{52}
$$

where $t_{\alpha/2, N-m}$ is the $t$ value obtained considering $\alpha$ as the significance level and $N-m$ as the degrees of freedom, then if $T_0$ falls outside this acceptance region, the null hypothesis $H_0: \Theta_j = 0$ is rejected. This implies that $\Theta_j$ is significant at the $\alpha$ level. Otherwise, if $T_0$ lies within the acceptance region, $\Theta_j$ is not significantly different from zero, and the null hypothesis cannot be rejected.

#### Penalty value based on the Derivative of the Sigmoid Linear Unit function

We propose a penalty value based on the derivative of the sigmoid function, defined as:

$$
\dot{\varsigma}(x(\varrho)) = \varsigma(x) [1 + (a(x - c))(1 - \varsigma(x))].
\tag{53}
$$

In this formulation, the parameters are defined as follows: $x$ has the dimension of $noV$; $c = noV / 2$; and $a$ is set as the ratio of the number of regressors in the current test model to $c$. This approach results in a distinct curve for each model, with the slope of the sigmoid curve becoming steeper as the number of regressors increases. The penalty value, $\varrho$, corresponds to the $y$ value of the sigmoid curve for the given number of regressors in $x$. Since the derivative of the sigmoid function can return negative values, we normalize $\varsigma$ as:

$$
\varrho = \varsigma - \mathrm{min}(\varsigma),
\tag{54}
$$

ensuring that $\varrho \in \mathbb{R}^{+}$.

However, two different models with the same number of regressors can yield significantly different results due to the varying importance of each regressor. To address this, we use the $t$-student test to assess the statistical relevance of each regressor and incorporate this information into the penalty function. Specifically, we calculate $n_{\Theta, H_{0}}$, the number of regressors that are not significant for the model. The penalty value is then adjusted based on the model size:

$$
\mathrm{model\_size} = n_{\Theta} + n_{\Theta, H_{0}}.
\tag{55}
$$

The objective function, which integrates the relative root mean squared error of the model with $\varrho$, is defined as:

$$
\mathcal{F} = \frac{\sqrt{\sum_{k=1}^{n} (y_k - \hat{y}_k)^2}}{\sqrt{\sum_{k=1}^{n} (y_k - \bar{y})^2}} \times \varrho.
\tag{56}
$$

This approach ensures that even if models have the same number of regressors, those with redundant regressors are penalized more heavily.

#### Case Studies: Simulation Results

In this section, six simulation examples are considered to illustrate the effectiveness of the MetaMSS algorithm. An analysis of the algorithm performance has been carried out considering different tuning parameters. The selected systems are generally used as a benchmark for model structures algorithms and were taken from [Wei, H. and Billings, S. A.]([Model structure selection using an integrated forward orthogonal search algorithm assisted by squared correlation and mutual information](https://www.inderscienceonline.com/doi/abs/10.1504/IJMIC.2008.020543)), [Falsone, A. and Piroddi, L. and Prandini, M.]([A randomized algorithm for nonlinear model structure selection](https://www.sciencedirect.com/science/article/abs/pii/S0005109815003088)), [Baldacchino, T. and Anderson, S. R. and Kadirkamanathan, V.]([Computational system identification for Bayesian NARMAX modelling](https://www.sciencedirect.com/science/article/abs/pii/S0005109813003063)), [Piroddi, L. and Spinelli, W.]([An identification algorithm for polynomial NARX models based on simulation error minimization](https://www.tandfonline.com/doi/abs/10.1080/00207170310001635419)), [Guo, Y. and Guo, L. Z. and Billings, S. A. and Wei, H.]([A New Iterative Orthogonal Forward Regression Algorithm](https://eprints.whiterose.ac.uk/107315/3/A%20New%20Iterative%20Orthogonal%20Forward%20Regression%20Algorithm%20-%20R2.pdf)), [Bonin, M. and Seghezza, V. and Piroddi, L.]([NARX model selection based on simulation error minimisation and LASSO](https://www.researchgate.net/publication/224153379_NARX_model_selection_based_on_simulation_error_minimisation_and_LASSO)), [Aguirre, L. A. and Barbosa, B. H. G. and Braga, A. P.]([Prediction and simulation errors in parameter estimation for nonlinear systems](https://www.sciencedirect.com/science/article/abs/pii/S0888327010001469)). Finally, a comparative analysis with respect to the [Randomized Model Structure Selection (RaMSS)]([A randomized algorithm for nonlinear model structure selection](https://www.sciencedirect.com/science/article/abs/pii/S0005109815003088)), the FROLS, and the [Reversible-jump Markov chain Monte Carlo]([Computational system identification for Bayesian NARMAX modelling](https://www.sciencedirect.com/science/article/abs/pii/S0005109813003063)) (RJMCMC)  algorithms has been accomplished to check out the goodness of the proposed method.

The simulation models are described as:

$$
\begin{align}
    & S_1: \quad y_k = -1.7y_{k-1} - 0.8y_{k-2} + x_{k-1} + 0.81x_{k-2} + e_k, \\
    & \qquad \quad \text{with } x_k \sim \mathcal{U}(-2, 2) \text{ and } e_k \sim \mathcal{N}(0, 0.01^2); \nonumber \\
    & S_2: \quad y_k = 0.8y_{k-1} + 0.4x_{k-1} + 0.4x_{k-1}^2 + 0.4x_{k-1}^3 + e_k, \\
    & \qquad \quad \text{with } x_k \sim \mathcal{N}(0, 0.3^2) \text{ and } e_k \sim \mathcal{N}(0, 0.01^2). \nonumber \\
    & S_3: \quad y_k = 0.2y_{k-1}^3 + 0.7y_{k-1}x_{k-1} + 0.6x_{k-2}^2 \nonumber \\
    &- 0.7y_{k-2}x_{k-2}^2 -0.5y_{k-2}+ e_k, \\
    & \qquad \quad \text{with } x_k \sim \mathcal{U}(-1, 1) \text{ and } e_k \sim \mathcal{N}(0, 0.01^2). \nonumber \\
    & S_4: \quad y_k = 0.7y_{k-1}x_{k-1} - 0.5y_{k-2} + 0.6x_{k-2}^2 \nonumber \\
    &- 0.7y_{k-2}x_{k-2}^2 + e_k, \\
    & \qquad \quad \text{with } x_k \sim \mathcal{U}(-1, 1) \text{ and } e_k \sim \mathcal{N}(0, 0.04^2). \nonumber \\
    & S_5: \quad y_k = 0.7y_{k-1}x_{k-1} - 0.5y_{k-2} + 0.6x_{k-2}^2 \nonumber \\
    &- 0.7y_{k-2}x_{k-2}^2 + 0.2e_{k-1} \nonumber \\
    & \qquad \quad - 0.3x_{k-1}e_{k-2} + e_k,\\
    & \qquad \quad \text{with } x_k \sim \mathcal{U}(-1, 1) \text{ and } e_k \sim \mathcal{N}(0, 0.02^2); \nonumber \\
    & S_6: \quad y_k = 0.75y_{k-2} + 0.25x_{k-2} - 0.2y_{k-2}x_{k-2} + e_k \nonumber \\
    & \qquad \quad \text{with } x_k \sim \mathcal{N}(0, 0.25^2) \text{ and } e_k \sim \mathcal{N}(0, 0.02^2); \nonumber
\end{align}
\tag{57}
$$

where $\mathcal{U}(a, b)$ are samples evenly distributed over~$[a, b]$, and $\mathcal{N}(\eta, \sigma^2)$ are samples with a Gaussian distribution with mean $\eta$ and standard deviation $\sigma$. All realizations of the systems are composed of a total of $500$ input-output data samples. Also, the same random seed is used to reproducibility purpose.

All tests shown in this section are based on the original implementation and are took from the results of my master thesis. At the time, the algorithm was performed in Matlab $2018$a environment, on a Dell Inspiron $5448$ Core i$5-5200$U CPU $2.20$GHz with $12$GB of RAM. However, it is not a hard task to adapt them to SysIdentPy.

Following the aforementioned studies, the maximum lags for the input and output are chosen to be, respectively, $n_u=n_y=4$ and the nonlinear degree is $\ell = 3$. The parameters related to the BPSOGSA are detailed on Table 8.

| Parameters | $n_u$ | $n_y$ | $\ell$ | p-value | max\_iter | n\_agents | $\alpha$ | $G_0$ |
|------------|-------|-------|--------|---------|-----------|-----------|----------|-------|
| Values     | $4$   | $4$   | $3$    | $0.05$  | $30$      | $10$      | $23$     | $100$ |
>Table 8. Parameters used in MetaMSS

$300$ runs of the Meta-MSS algorithm have been executed for each model, aiming to compare some statistics about the algorithm performance. The elapsed time, the time required to obtain the final model, and correctness, the percentage of exact model selections, are analyzed.

The results in Table 9 are obtained with the parameters configured accordingly to Table 8.

|                     | $S_1$ | $S_2$ | $S_3$ | $S_4$ | $S_5$ | $S_6$ |
| ------------------- | ----- | ----- | ----- | ----- | ----- | ----- |
| Correct model       | 100\% | 100\% | 100\% | 100\% | 100\% | 100\% |
| Elapsed time (mean) | 5.16s | 3.90s | 3.40s | 2.37s | 1.40s | 3.80s |
>Table 9. Overall performance of the MetaMSS

Table 9 shows that all the model terms are correctly selected using the Meta-MSS. It is worth to notice that even the model $S_5$, which have an autoregressive noise, was correctly selected using the proposed algorithm. This result resides in the evaluation of all regressors individually, and the ones considered redundant are removed from the model.

Figure 15 presents the convergence of each execution of Meta-MSS. It is noticeable that the majority of executions converges to the correct model structures with $10$ or fewer iterations. The reason for this relies on the maximum number of iterations and the number of search agents. The first one is related to the acceleration coefficient, which boosts the exploration phase of the algorithm, while the latter increases the number of candidate models to be evaluated. Intuitively, one can see that both parameters influence the elapsed time and, more importantly, the model structure selected to compose the final model. Consequently, an inappropriate choice of one of them may results in sub/over-parameterized models, since the algorithm can converge to a local optimum. The next subsection presents an analysis of the max\_iter and n\_agents influence in the algorithm performance.

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/metamss_convergence.png?raw=true)
> Figure 15. Convergence of Meta-MSS for different model structures. The figure illustrates the convergence behavior of the Meta-MSS algorithm across multiple executions. Each curve represents the convergence trajectory for a specific model structure from $S_1$ to $S_6$ over a maximum of 30 iterations.

#### Influence of the $max\_iter$ and $n\_agents$ parameters

The simulation models are used to show the performance of the Meta-MSS considering different tuning for `max_iter` and `n_agents` parameters. First, we set and uphold the `max_iter=30` while the `n_agents` are changed. Then, we set and uphold the `n_agents` while the `max_iter` is modified. The results detailed in this section have been obtained by setting the remaining parameters according to Table 8.

|                                    |                         | $S_1$     | $S_2$     | $S_3$   | $S_4$     | $S_5$     | $S_6$     |
| ---------------------------------- | ----------------------- | --------- | --------- | ------- | --------- | --------- | --------- |
| **max\_iter = 30, n\_agents = 1**  | **Correct model**       | $65\%$    | $55.66\%$ | $14\%$  | $14\%$    | $7.3\%$   | $20.66\%$ |
|                                    | **Elapsed time (mean)** | $0.26$s   | $0.19$s   | $0.15$s | $0.11$s   | $0.13$s   | $0.13$s   |
| **max\_iter = 30, n\_agents = 5**  | **Correct model**       | $100\%$   | $100\%$   | $99\%$  | $98\%$    | $91.66\%$ | $98.33\%$ |
|                                    | **Elapsed time (mean)** | $2.08$s   | $1.51$s   | $1.41$s | $0.99$s   | $0.59$s   | $1.13$s   |
| **max\_iter = 30, n\_agents = 20** | **Correct model**       | $100\%$   | $100\%$   | $100\%$ | $100\%$   | $100\%$   | $100\%$   |
|                                    | **Elapsed time (mean)** | $12.88$s  | $9.10$s   | $8.77$s | $5.70$s   | $3.37$s   | $9.50$s   |
| **max\_iter = 5, n\_agents = 10**  | **Correct model**       | $96.33\%$ | $99\%$    | $86\%$  | $93.66\%$ | $93\%$    | $97.33\%$ |
|                                    | **Elapsed time (mean)** | $0.92$s   | $0.73$s   | $0.72$s | $0.52$s   | $0.29$s   | $0.64$s   |
| **max\_iter = 15, n\_agents = 10** | **Correct model**       | $100\%$   | $100\%$   | $99\%$  | $99\%$    | $100\%$   | $100\%$   |
|                                    | **Elapsed time (mean)** | $2.80$s   | $2.33$s   | $2.25$s | $1.60$s   | $0.90$s   | $2.30$s   |
| **max\_iter = 50, n\_agents = 10** | **Correct model**       | $100\%$   | $100\%$   | $100\%$ | $100\%$   | $100\%$   | $100\%$   |
|                                    | **Elapsed time (mean)** | $7.38$s   | $5.44$s   | $4.56$s | $3.01$s   | $2.10$s   | $4.52$s   |
>Table 10.

The aggregated results in Table 10 confirms the expected performance regarding the elapsed time and percentage of correct models. Indeed, both metrics increases significantly as the number of agents and the maximum number of iteration increases. The number of agents is very relevant because it yields a broader exploration of the search space. All system are affected by the increase in the number of agents and the maximum number of iterations.

Regarding all tested systems, it is straightforward to notice that the more extensive exploration dramatically impacts on the exactitude of the selection procedure. If only a few agents are assigned, the performance of Meta-MSS algorithm deteriorates significantly, especially for systems $S_3, S_4$ and $S_5$. The maximum number of iteration empowers agents to explore, globally and locally, the space around the candidate models tested so far. In this sense, as the number of iterations increases, more the agents can explore the search space and examine different regressors.

If these parameters are improperly chosen, the algorithm might fail to select the best model structure. In this respect, the results presented here concerns only the selected systems. The larger the search space, the larger the number of agents and iterations should be. Although the computational effort increases with larger values for n\_agents and max\_iteration, the algorithm remains very efficient regarding the elapsed time for all tuning configurations that ensured the selection of the exact model structures.

#### Selection of over and sub-parameterized models

Regardless of the successful selection of all models structures by the Meta-Structure Selection Algorithm, one can ask how the models differs from the true ones in the cases presented in Table 10 where the algorithm failed to ensure $100\%$ of correctness. Figure 16 depicts the distribution of terms number selected in each case. It is evident that the number of over-parameterized models selected is higher than the sub-parameterized in overall. Regarding the cases where the number of search agents are low, due to low exploration and exploitation capacity, the algorithm converged early and resulted in models with a high number of spurious regressors. In respect to $S_2$ and $S_5$, for example, with n\_agents$=1$, the algorithm ends up selecting models with more than $20$ terms. One can say this was a extreme scenario for comparison purpose. However, a suitable choice for the parameters is intrinsically related to the dimension of the search space. Referring to cases where n\_agents$\geq 5$, the number of spurious terms decreased significantly where the algorithm failed to select the true models.

Furthermore, it is interesting to point out the importance of tuning the parameters properly because since the exploration and exploitation phase of the algorithm are strongly dependent on them. A premature convergence of the algorithm may result in models with the factual number of terms, but with wrong ones. This happened with all cases with `n_agents=1`. For example, the algorithm generates models with correct number of terms in $33.33\%$ of the cases analyzed regarding $S_3$. However, Table 10 shows that only $14\%$ are, in fact, equivalent to the true model.

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/metamss_terms_distribution.png?raw=true)
> Figure 16. The distribution of terms number selected for each simulated models concerning the variation of the `max_iter` and `n_agents`.

#### Selection of over and sub-parameterized models

Regardless of the successful selection of all models structures by the MetaMSS, one can ask how the models differ from the true ones in the cases presented in Table 10 where the algorithm failed to ensure $100\%$ of correctness. Figure 16 depicts the distribution of terms number selected in each case. It is evident that the number of over-parameterized models selected is higher than the sub-parameterized in overall. Regarding the cases where the number of search agents is low, due to low exploration and exploitation capacity, the algorithm converged early and resulted in models with a high number of spurious regressors. In respect to $S_2$ and $S_5$, for example, with `n_agents=1`, the algorithm ends up selecting models with more than $20$ terms. One can say this was an extreme scenario for comparison purpose. However, a suitable choice for the parameters is intrinsically related to the dimension of the search space. By referring to cases where `n_agents`$\geq 5$, the number of spurious terms decreased significantly where the algorithm failed to select the true models.

Furthermore, it is interesting to point out the importance of tuning the parameters properly because since the exploration and exploitation phase of the algorithm is strongly dependent on them. A premature convergence of the algorithm may result in models with the actual number of terms, but with wrong ones. This issue happened with all cases with `n_agents=1`. For example, the algorithm generates models with the correct number of terms in $33.33\%$ of the cases analyzed regarding $S_3$. However, Table 10 shows that only $14\%$ are, in fact, equivalent to the true model.

The systems $S_1$, $S_2$, $S_3$, $S_4$ and $S_6$ has been used as benchmark by [Bianchi, F., Falsone, A., Prandini, M. and Piroddi, L.](https://www.tandfonline.com/doi/abs/10.1080/00207721.2016.1244309), so we can compare directly our results with those reported by the author in his thesis. All techniques used $n_y=n_u=4$ and $\ell = 3$. The RaMSS and the RaMSS with Conditional Linear Family (C-RaMSS) used the following configuration for the tuning parameters: $K=1$, $\alpha = 0.997$, $NP = 200$ and $v=0.1$. The Meta-Structure Selection Algorithm was tuned according to Table 8.

|                     |                         | $S_1$     | $S_2$    | $S_3$    | $S_4$    | $S_6$    |
| ------------------- | ----------------------- | --------- | -------- | -------- | -------- | -------- |
| **Meta-MSS**        | **Correct model**       | $100\%$   | $100\%$  | $100\%$  | $100\%$  | $100\%$  |
|                     | **Elapsed time (mean)** | $5.16$s   | $3.90$s  | $3.40$s  | $2.37$s  | $3.80$s  |
| **RaMSS- $NP=100$** | **Correct model**       | $90.33\%$ | $100\%$  | $100\%$  | $100\%$  | $66\%$   |
|                     | **Elapsed time (mean)** | $3.27$s   | $1.24$s  | $2.59$s  | $1.67$s  | $6.66$s  |
| **RaMSS- $NP=200$** | **Correct model**       | $78.33\%$ | $100\%$  | $100\%$  | $100\%$  | $82\%$   |
|                     | **Elapsed time (mean)** | $6.25$s   | $2.07$s  | $4.42$s  | $2.77$s  | $9.16$s  |
| **C-RaMSS**         | **Correct model**       | $93.33\%$ | $100\%$  | $100\%$  | $100\%$  | $100\%$  |
|                     | **Elapsed time (mean)** | $18$s     | $10.50$s | $16.96$s | $10.56$s | $48.52$s |
> Table 11. Comparative analysis between MetaMSS, RaMSS, and C-RaMSS

In terms of correctness, the MetaMSS outperforms (or at least equals) the RaMSS and C-RaMSS for all analyzed systems as shown in Table 11. Regarding $S_6$, the correctness rate increased by $18\%$ when compared with RaMSS and the elapsed time required for C-RaMSS obtain $100\%$ of correctness is $1276.84\%$ higher than the MetaMSS. Furthermore, the MetaMSS is notably more computationally efficient than C-RaMSS and similar to RaMSS.

#### MetaMSS vs FROLS

The FROLS algorithm was applied to all tested systems, with the results summarized in Table 12. The algorithm successfully selected the correct model terms for $S_2$ and $S_6$. However, it failed to identify two out of four regressors for $S_1$. For $S_3$, FROLS included $y_{k-1}$ instead of the correct term $y_{k-1}^3$. Similarly, $S_4$ incorrectly included $y_{k-4}$ rather than the required term $y_{k-2}$. Additionally, for $S_5$, the algorithm produced an incorrect model structure by including the spurious term $y_{k-4}$.

|                       | Meta-MSS Regressor  | Correct | FROLS Regressor  | Correct |
|-----------------------|--------------------|---------|------------------|---------|
| **$S_1$**             | $y_{k-1}$          | yes     | $y_{k-1}$        | yes     |
|                       | $y_{k-2}$          | yes     | $y_{k-4}$        | no      |
|                       | $x_{k-1}$          | yes     | $x_{k-1}$        | yes     |
|                       | $x_{k-2}$          | yes     | $x_{k-4}$        | no      |
| **$S_2$**             | $y_{k-1}$          | yes     | $y_{k-1}$        | yes     |
|                       | $x_{k-1}$          | yes     | $x_{k-1}$        | yes     |
|                       | $x_{k-1}^2$        | yes     | $x_{k-1}^2$      | yes     |
|                       | $x_{k-1}^3$        | yes     | $x_{k-1}^3$      | yes     |
| **$S_3$**             | $y_{k-1}^3$        | yes     | $y_{k-1}$        | no      |
|                       | $y_{k-1}x_{k-1}$   | yes     | $y_{k-1}x_{k-1}$ | yes     |
|                       | $x_{k-2}^2$        | yes     | $x_{k-2}^2$      | yes     |
|                       | $y_{k-2}x_{k-2}^2$ | yes     | $y_{k-2}x_{k-2}^2$ | yes   |
|                       | $y_{k-2}$          | yes     | $y_{k-2}$        | yes     |
| **$S_4$**             | $y_{k-1}x_{k-1}$   | yes     | $y_{k-1}x_{k-1}$ | yes     |
|                       | $y_{k-2}$          | yes     | $y_{k-4}$        | no      |
|                       | $x_{k-2}^2$        | yes     | $x_{k-2}^2$      | yes     |
|                       | $y_{k-2}x_{k-2}^2$ | yes     | $y_{k-2}x_{k-2}^2$ | yes   |
| **$S_5$**             | $y_{k-1}x_{k-1}$   | yes     | $y_{k-1}x_{k-1}$ | yes     |
|                       | $y_{k-2}$          | yes     | $y_{k-4}$        | no      |
|                       | $x_{k-2}^2$        | yes     | $x_{k-2}^2$      | yes     |
|                       | $y_{k-2}x_{k-2}^2$ | yes     | $y_{k-2}x_{k-2}^2$ | yes   |
| **$S_6$**             | $y_{k-2}$          | yes     | $y_{k-2}$        | yes     |
|                       | $x_{k-1}$          | yes     | $x_{k-1}$        | yes     |
|                       | $y_{k-2}x_{k-2}$   | yes     | $y_{k-2}x_{k-1}$ | yes     |
> Table 12. Comparative analysis between MetaMSS and FROLS

#### Meta-MSS vs RJMCMC

The $S_4$ model is taken from Baldacchino, Anderson, and Kadirkamanathan's work ([Computational System Identification for Bayesian NARMAX Modelling](https://www.sciencedirect.com/science/article/abs/pii/S0005109813003063)). In their study, the maximum lags for the input and output are $n_y = n_u = 4$, and the nonlinear degree is $\ell = 3$. The authors ran the RJMCMC algorithm 10 times on the same input-output data. The RJMCMC method successfully identified the true model structure 7 out of 10 times. In contrast, the MetaMSS algorithm consistently identified the true model structure in all runs. These results are summarized in Table 13.

Additionally, the RJMCMC method has notable drawbacks that are addressed by the MetaMSS algorithm. Specifically, RJMCMC is computationally intensive, requiring $30,000$ iterations to achieve results. Furthermore, it relies on various probability distributions to simplify the parameter estimation process, which can complicate the computations. In contrast, MetaMSS offers a more efficient and straightforward approach, avoiding these issues.

|                  | Meta-MSS Model       | Correct | RJMCMC Model 1 ($7\times$) | RJMCMC Model 2      | RJMCMC Model 3      | RJMCMC Model 4      | Correct |
|------------------|----------------------|---------|---------------------------|---------------------|---------------------|---------------------|---------|
| **$S_4$**        | $y_{k-1}x_{k-1}$      | yes     | $y_{k-1}x_{k-1}$           | $y_{k-1}x_{k-1}$    | $y_{k-1}x_{k-1}$    | $y_{k-1}x_{k-1}$    | yes     |
|                  | $y_{k-2}$             | yes     | $y_{k-2}$                  | $y_{k-2}$           | $y_{k-2}$           | $y_{k-2}$           | yes     |
|                  | $x_{k-2}^2$           | yes     | $x_{k-2}^2$                | $x_{k-2}^2$         | $x_{k-2}^2$         | $x_{k-2}^2$         | yes     |
|                  | $y_{k-2}x_{k-2}^2$    | yes     | $y_{k-2}x_{k-2}^2$         | $y_{k-2}x_{k-2}^2$  | $y_{k-2}x_{k-2}^2$  | $y_{k-2}x_{k-2}^2$  | yes     |
|                  | -                      | -       | -                           | $y_{k-3}x_{k-3}$    | $x_{k-4}^2$         | $x_{k-1}x_{k-3}^2$  | no      |
> Table 13. Comparative analysis between MetaMSS and RJMCMC.

### MetaMSS algorithm using SysIdentPy

Consider the same data used in the Overview of the Information Criteria Methods.

```python
from sysidentpy.model_structure_selection import MetaMSS


basis_function = Polynomial(degree=2)
model = MetaMSS(
    ylag=2,
    xlag=2,
    random_state=42,
    basis_function=basis_function,
)

model.fit(X=x_train, y=y_train)
yhat = model.predict(X=x_valid, y=y_valid)

r = pd.DataFrame(
    results(
        model.final_model,
        model.theta,
        model.err,
        model.n_terms,
        err_precision=8,
        dtype="sci",
    ),
    columns=["Regressors", "Parameters", "ERR"],
)
print(r)
plot_results(y=y_valid, yhat=yhat, n=1000)
```

The MetaMSS algorithm does not rely on information criteria methods such as ERR for model structure selection, which is why it does not involve those hyperparameters. This is also true for the AOLS and ER algorithms. For more details on how to use these methods and their associated hyperparameters, please refer to the documentation.

When it comes to parameter estimation, SysIdentPy allows the use of any available method, regardless of the model structure selection algorithm. Users can select from a range of parameter estimation methods to apply to their chosen model structure. This flexibility enables users to explore various modeling approaches and customize their system identification process. While the examples provided use the default parameter estimation method, users are encouraged to experiment with different options to find the best fit for their needs.

The results of the MetaMSS are

| Regressors | Parameters  | ERR           |
|------------|-------------|---------------|
| y(k-1)     | 1.8004E-01  | 0.00000000E+00|
| x1(k-2)    | 8.9747E-01  | 0.00000000E+00|

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/metamss_result_c4_example.png?raw=true)
> Figure 17. Free Run Simulation for the model fitted using MetaMSS.

The `results` method brings ERR as 0 for every regressor because, as mentioned, ERR algorithm is not executed in this case.

## Accelerated Orthogonal Least Squares (AOLS) and Entropic Regression (ER)

In addition to FROLS and MetaMSS, SysIdentPy includes two other methods for model structure selection: Accelerated Orthogonal Least Squares (AOLS) and Entropic Regression (ER). While I won't delve into the details of these methods in this section as I have with FROLS and MetaMSS, I will provide an overview and references for further reading:

- **Accelerated Orthogonal Least Squares (AOLS):** For an in-depth exploration of AOLS, refer to the original paper [here](https://www.sciencedirect.com/science/article/abs/pii/S1051200418305311).
- **Entropic Regression (ER):** Detailed information about ER can be found in the original paper [here](https://arxiv.org/pdf/1905.08061).

For now, I will demonstrate how to use these methods within SysIdentPy.

### Accelerated Orthogonal Least Squares

```python
from sysidentpy.model_structure_selection import AOLS

basis_function = Polynomial(degree=2)
model = AOLS(
    ylag=2,
    xlag=2,
    basis_function=basis_function,
)

model.fit(X=x_train, y=y_train)
yhat = model.predict(X=x_valid, y=y_valid)

r = pd.DataFrame(
    results(
        model.final_model,
        model.theta,
        model.err,
        model.n_terms,
        err_precision=8,
        dtype="sci",
    ),
    columns=["Regressors", "Parameters", "ERR"],
)
print(r)
plot_results(y=y_valid, yhat=yhat, n=1000)
```

| Regressors | Parameters  | ERR           |
|------------|-------------|---------------|
| x1(k-2)    | 9.1542E-01  | 0.00000000E+00|

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/aols_example_c4.png?raw=true)
> Figure 18. Free Run Simulation for the model fitted using AOLS algorithm.

### Entropic Regression

```python

from sysidentpy.model_structure_selection import ER

basis_function = Polynomial(degree=2)
model = ER(
    ylag=2,
    xlag=2,
    basis_function=basis_function,
)
model.fit(X=x_train, y=y_train)
yhat = model.predict(X=x_valid, y=y_valid)

r = pd.DataFrame(
    results(
        model.final_model,
        model.theta,
        model.err,
        model.n_terms,
        err_precision=8,
        dtype="sci",
    ),
    columns=["Regressors", "Parameters", "ERR"],
)

print(r)
plot_results(y=y_valid, yhat=yhat, n=1000)
```

| Regressors | Parameters  | ERR           |
|------------|-------------|---------------|
| 1          | -2.4554E-02 | 0.00000000E+00|
| x1(k-2)    | 9.0273E-01  | 0.00000000E+00|

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/er_example_c4.png?raw=true)
> Figure 19. Free Run Simulation for the model fitted using Entropic Regression algorithm.
