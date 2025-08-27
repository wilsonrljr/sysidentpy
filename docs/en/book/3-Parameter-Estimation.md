## Least Squares

Consider the NARX model described in a generic form as

$$
\begin{equation}
    y_k = \psi^\top_{k-1}\hat{\Theta} + \xi_k,
\end{equation}
\tag{3.1}
$$

where $\psi^\top_{k-1} \in \mathbb{R}^{n_r \times n}$ is the information matrix, also known as the regressors' matrix. The information matrix is the input and output transformation based in a basis function and $\hat{\Theta}~\in \mathbb{R}^{n_{\Theta}}$ the vector of estimated parameters. The model above can also be represented in a matrix form as:

$$
\begin{equation}
    y = \Psi\hat{\Theta} + \Xi,
\end{equation}
\tag{3.2}
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
\tag{3.3}
$$


> We will consider the polynomial basis function to keep the examples straightforward, but the methods here will work for any other basis function.

The parametric NARX model is linear in the parameters $\Theta$, so we can use well known algorithms, like the linear Least Squares algorithm developed by Gauss in $1795$, to estimate the model parameters. The idea is to find the parameter vector that minimizes the $l2$-norm, also known as the residual sum of squares, described as

$$
\begin{equation}
    J_{\hat{\Theta}} = \Xi^\top \Xi = (y - \Psi\hat{\Theta})^\top(y - \Psi\hat{\Theta}) = \lVert y - \Psi\hat{\Theta} \rVert^2.
\end{equation}
\tag{3.4}
$$

In Equation 3.4 , $\Psi\hat{\Theta}$ is the one-step ahead prediction of $y_k$, expressed as

$$
\begin{equation}
    \hat{y}_{1_k} = g(y_{k-1}, u_{k-1}\lvert ~\Theta),
\end{equation}
\tag{3.5}
$$

where $g$ is some unknown polynomial function. If the gradient of $J_{\Theta}$ with respect to $\Theta$ is equal to zero, then we have the normal equation and the Least Squares estimate is expressed as

$$
\begin{equation}
    \hat{\Theta}  = (\Psi^\top\Psi)^{-1}\Psi^\top y,
\end{equation}
\tag{3.6}
$$

where $(\Psi^\top\Psi)^{-1}\Psi^\top$ is called the pseudo-inverse of the matrix $\Psi$, denoted $\Psi^+ \in \mathbb{R}^{n \times n_r}$.

In order to have a bias-free estimator, the following are the basic assumptions needed for the least-squares method:
- A1 - There is no correlation between the error vector, $\Xi$, and the matrix of regressors, $\Psi$. Mathematically:
- $\mathrm{E}\{[(\Psi^\top\Psi)^{-1}\Psi^\top] \Xi\} = \mathrm{E}[(\Psi^\top\Psi)^{-1}\Psi^\top] \mathrm{E}[\Xi]; \tag{3.7}$
- A2 - The error vector $\Xi$ is a zero mean white noise sequence:
- $\mathrm{E}[\Xi] = 0; \tag{3.8}$
- A3 - The covariance matrix of the error vector is
- $\mathrm{Cov}[\hat{\Theta}] = \mathrm{E}[(\Theta - \hat{\Theta})(\Theta - \hat{\Theta})^\top] = \sigma^2(\Psi^\top\Psi); \tag{3.9}$
- A4 - The matrix of regressors, $\Psi$, is full rank.

The aforementioned assumptions are needed to guarantee that the Least Squares algorithm produce a unbiased final model.

#### Example

Let's see a practical example. Consider the model

$$
y_k = 0.2y_{k-1} + 0.1y_{k-1}x_{k-1} + 0.9x_{k-2} + e_{k}
\tag{3.10}
$$

We can generate the input `X`  and output `y` using SysIdentPy. Before getting in the details, let run a simple model using SysIdentPy. Because we know a priori that the system we are trying to have is not linear (the simulated system have an interaction term $0.1y_{k-1}x_{k-1}$) and the order is 2 (the maximum lag of the input and output), we will set the hyperparameters accordingly. Note that this a simulated scenario, and you'll not have such information a priori in a real identification task. But don't worry, the idea, for now, is just show how things works and we will develop some real models along the book.

```python
from sysidentpy.utils.generate_data import get_siso_data
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial
from sysidentpy.parameter_estimation import LeastSquares
from sysidentpy.utils.display_results import results

x_train, x_test, y_train, y_test = get_siso_data(
    n=1000, colored_noise=False, sigma=0.001, train_percentage=90
)

basis_function = Polynomial(degree=2)
estimator = LeastSquares()
model = FROLS(
    n_info_values=3,
    ylag=1,
    xlag=2,
    estimator=estimator,
    basis_function=basis_function,
)
model.fit(X=x_train, y=y_train)
# print the identified model
r = pd.DataFrame(
    results(
        model.final_model,
        model.theta,
        model.err,
        model.n_terms
    ),
    columns=["Regressors", "Parameters", "ERR"],
)
print(r)

Regressors   Parameters             ERR
0        x1(k-2)  9.0001E-01  9.56885108E-01
1         y(k-1)  2.0000E-01  3.96313039E-02
2  x1(k-1)y(k-1)  1.0001E-01  3.48355000E-03
```

As you can see, the final model have the same 3 regressors of the simulated system and the parameters are very close the ones used to simulate the system. This shows us that the Least Squares performed well for this data.

In this example, however, we are applying a Model Structure Selection algorithm (FROLS), which we will see in chapter 6. That's why the final model have only 3 regressors. The parameter estimation algorithm do not choose which terms to include in the model, so if we have a expanded basis function with 6 regressors, it will estimate the parameter for each one of the regressors.

To check how this work, we can use SysIdentPy without Model Structure Selection by generating the information matrix and applying the parameter estimation algorithm directly.

```python
from sysidentpy.utils.generate_data import get_siso_data
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial
from sysidentpy.parameter_estimation import LeastSquares
from sysidentpy.utils import build_lagged_matrix

x_train, x_test, y_train, y_test = get_siso_data(
    n=1000, colored_noise=False, sigma=0.001, train_percentage=90
)
xlag = 2
ylag = 2
max_lag = 2
regressor_matrix = build_lagged_matrix(
    x=x_train, y=y_train, xlag=xlag, ylag=ylag, model_type="NARMAX",
)
# apply the basis function
psi = Polynomial(degree=2).fit(regressor_matrix, max_lag=max_lag, xlag=xlag, ylag=ylag)
theta = LeastSquares().optimize(psi, y_train[max_lag:, :])
theta

[[-4.1511e-06]
 [ 2.0002e-01]
 [ 1.1237e-05]
 [ 1.0068e-05]
 [ 8.9997e-01]
 [-6.3216e-05]
 [ 1.3298e-04]
 [ 1.0008e-01]
 [ 6.3118e-05]
 [-5.6031e-05]
 [-1.9073e-05]
 [-1.8223e-04]
 [ 1.1307e-04]
 [-1.6601e-04]
 [-8.5068e-05]]
```

In this case, we have 15 model parameters. If we take a look in the basis function expansion where the degree of the polynomial is equal to 2 and the lags for `y` and `x` are set to 2, we have

```python
from sysidentpy.utils.narmax_tools import regressor_code
basis_function = Polynomial(degree=2)
regressors = regressor_code(
    X=x_train,
    xlag=2,
    ylag=2,
    model_type="NARMAX",
    model_representation="Polynomial",
    basis_function=basis_function,
)
regressors

array([[ 0, 0],
   [1001, 0],
   [1002, 0],
   [2001, 0],
   [2002, 0],
   [1001, 1001],
   [1002, 1001],
   [2001, 1001],
   [2002, 1001],
   [1002, 1002],
   [2001, 1002],
   [2002, 1002],
   [2001, 2001],
   [2002, 2001],
   [2002, 2002]]
   )
```

The regressors is how SysIdentPy encode the polynomial basis function following this  codification pattern:

- $0$ is the constant term,\n",
- $[1001] = y_{k-1}$
- $[100n] = y_{k-n}$
- $[200n] = x1_{k-n}$
- $[300n] = x2_{k-n}$
- $[1011, 1001] = y_{k-11} \\times y_{k-1}$
- $[100n, 100m] = y_{k-n} \times y_{k-m}$
- $[12001, 1003, 1001] = x11_{k-1} \times y_{k-3} \times y_{k-1}$,
- and so on

 So, if you take a look at the parameters, we can see that the Least Squares algorithm estimation for the terms that belongs to the simulated system are very close to the real values.

```python
[1001, 0] -> [ 2.00002486e-01]
[2002, 0] -> [ 8.99927332e-01]
[2001, 1001] -> [ 1.00062340e-01]
```

Moreover, the parameters estimated for the other regressors are considerably lower values than the ones estimated for the correct terms, indicating that the other might not be relevant to the model.

You can start thinking that we only need to define a basis function and apply some parameter estimation technique to build NARMAX models. However, as mentioned before, the main goal of the NARMAX methods is to build the best model possible while keeping it simple. And that's true for the case where we applied the FROLS algorithm. Besides, when dealing with system identification we want to recover the dynamics of the system under study, so adding more terms than necessary can lead to unexpected behaviors, poor performance and unstable models. Remember, this is only a toy example, so in real cases the model structure selection is fundamental.

You can implement Least Squares method as simple as

```python
import numpy as np

def simple_least_squares(psi, y):
    return np.linalg.pinv(psi.T @ psi) @ psi.T @ y

# use the psi and y data created in previous examples or
# create them again here to run the example.
theta = simple_least_squares(psi, y_train[max_lag:, :])

theta

array(
	[
	   [-1.08377785e-05],
	   [ 2.00002486e-01],
	   [ 1.73422294e-05],
	   [-3.50957931e-06],
	   [ 8.99927332e-01],
	   [ 2.04427279e-05],
	   [-1.47542408e-04],
	   [ 1.00062340e-01],
	   [ 4.53379771e-05],
	   [ 8.90006341e-05],
	   [ 1.15234873e-04],
	   [ 1.57770755e-04],
	   [ 1.58414037e-04],
	   [-3.09236444e-05],
	   [-1.60377753e-04]
	]
)
```

As you can see, the estimated parameters are very close. However, be careful when using such approach in under-, well-, or over-determined systems. We recommend to use the numpy or scipy `lstsq` methods.

## Total Least Squares

This section is based on the [Markovsky, I., & Van Huffel, S. (2007). Overview of total least squares methods. Signal Processing.](https://people.duke.edu/~hpgavin/SystemID/References/Markovsky+VanHuffel-SP-2007.pdf).

The Total Least Squares (TLS) algorithm, is a statistical method used to find the best-fitting linear relationship between variables when both the input and output signals present white noise perturbation. Unlike ordinary least squares (OLS), which assumes that only the dependent variable is subject to error, TLS considers errors in all measured variables, providing a more robust solution in many practical applications. The algorithm was proposed by Golub and Van Loan.

In TLS, we assume errors in both $\mathbf{X}$ and $\mathbf{Y}$, denoted as $\Delta \mathbf{X}$ and $\Delta \mathbf{Y}$, respectively. The true model becomes:

$$
\mathbf{Y} + \Delta \mathbf{Y} = (\mathbf{X} + \Delta \mathbf{X}) \mathbf{B}
\tag{3.11}
$$

Rearranging, we get:

$$
\Delta \mathbf{Y} = \Delta \mathbf{X} \mathbf{B}
\tag{3.12}
$$

### Objective Function

The TLS solution minimizes the Frobenius norm of the total perturbations in $\mathbf{X}$ and $\mathbf{Y}$:

$$
\min_{\Delta \mathbf{X}, \Delta \mathbf{Y}} \|[\Delta \mathbf{X}, \Delta \mathbf{Y}]\|_F
\tag{3.13}
$$

subject to:

$$
(\mathbf{X} + \Delta \mathbf{X}) \mathbf{B} = \mathbf{Y} + \Delta \mathbf{Y}
\tag{3.14}
$$

where $\| \cdot \|_F$ denotes the Frobenius norm.

### Classical Solution

The classical approach to solve the TLS problem is by using Singular Value Decomposition (SVD). The augmented matrix $[\mathbf{X}, \mathbf{Y}]$ is decomposed as:

$$
[\mathbf{X}, \mathbf{Y}] = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T
\tag{3.15}
$$

where $\mathbf{U}$ is an $n \times n$ orthogonal matrix, $\Sigma=\operatorname{diag}\left(\sigma_1, \ldots, \sigma_{n+d}\right)$ is a diagonal matrix of singular values; and $\mathbf{V}$ is an orthogonal matrix defined as

$$
V:=\left[\begin{array}{cc}
V_{11} & V_{12} \\
V_{21} & V_{22}
\end{array}\right] \quad \begin{aligned}
\end{aligned} \quad \text { and } \quad \Sigma:=\left[\begin{array}{cc}
\Sigma_1 & 0 \\
0 & \Sigma_2
\end{array}\right] \begin{gathered}
\end{gathered} .
\tag{3.16}
$$

A total least squares solution exists if and only if $V_{22}$ is non-singular. In addition, it is unique if and only if $\sigma_n \neq \sigma_{n+1}$. In the case when the total least squares solution exists and is unique, it is given by

$$
\widehat{X}_{\mathrm{tls}}=-V_{12} V_{22}^{-1}
\tag{3.17}
$$

and the corresponding total least squares correction matrix is

$$
\Delta C_{\mathrm{tls}}:=\left[\begin{array}{ll}
\Delta A_{\mathrm{tls}} & \Delta B_{\mathrm{tls}}
\end{array}\right]=-U \operatorname{diag}\left(0, \Sigma_2\right) V^{\top} .
\tag{3.18}
$$

This is implemented in SysIdentPy as follows:

```python
def optimize(self, psi: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Estimate the model parameters using Total Least Squares method.

    Parameters
    ----------
    psi : ndarray of floats
        The information matrix of the model.
    y : array-like of shape = y_training
        The data used to training the model.

    Returns
    -------
    theta : array-like of shape = number_of_model_elements
        The estimated parameters of the model.

    """
    check_linear_dependence_rows(psi)
    full = np.hstack((psi, y))
    n = psi.shape[1]
    _, _, v = np.linalg.svd(full, full_matrices=True)
    theta = -v.T[:n, n:] / v.T[n:, n:]
    return theta.reshape(-1, 1)
```

To use it in the modeling task, just import it like we did in the Least Squares example.

From now on the examples will not include the Model Structure Selection step. The goal here is to focus on the parameter estimation methods. However, we already provided an example including MSS in the Least Squares section, so you will not have any problem to test that with other parameter estimation algorithms.

```python
from sysidentpy.utils.generate_data import get_siso_data
from sysidentpy.basis_function import Polynomial
from sysidentpy.parameter_estimation import TotalLeastSquares
from sysidentpy.utils import build_lagged_matrix

x_train, x_test, y_train, y_test = get_siso_data(
    n=1000, colored_noise=False, sigma=0.001, train_percentage=90
)
xlag = 2
ylag = 2
max_lag = 2
regressor_matrix = build_lagged_matrix(
    x=x_train, y=y_train, xlag=xlag, ylag=ylag, model_type="NARMAX",
)
# apply the basis function
psi = Polynomial(degree=2).fit(regressor_matrix, max_lag=max_lag, xlag=xlag, ylag=ylag)
theta = TotalLeastSquares().optimize(psi, y_train[max_lag:, :])
theta

[[ 1.3321e-04]
 [ 2.0014e-01]
 [-1.1771e-04]
 [ 5.8085e-05]
 [ 9.0011e-01]
 [-1.5490e-04]
 [-1.3517e-05]
 [ 9.9824e-02]
 [ 8.2326e-05]
 [-2.2814e-04]
 [-7.0837e-05]
 [-5.4319e-05]
 [-1.7472e-04]
 [-2.0396e-04]
 [ 1.7416e-05]]
```

## Recursive Least Squares

Consider the regression model

$$ y_k = \mathbf{\Psi}_k^T \theta_k + \epsilon_k \tag{3.19}$$

where:
- $y_k$ is the observed output at time $ k $.
- $\mathbf{\Psi}_k$ is the information matrix at time $k$.
- $\theta_k$ is the parameter vector to be estimated at time $k$.
- $\epsilon_k$ is the noise at time $k$.

The Recursive Least Squares (RLS) algorithm updates the parameter estimate $\theta_k$ recursively as new data points $(\mathbf{x}_k, y_k)$ become available, minimizing a weighted linear least squares cost function relating to the information matrix in a sequential manner. RLS is particularly useful in real-time applications where the data arrives sequentially and the model needs continuous updating or for modeling time varying systems (if the forgetting factor is included).

Because it's a recursive estimation, it is useful to relate $\hat{\Theta}_k$ to $\hat{\Theta}_{k-1}$. In other words, the new $\hat{\Theta}_k$ depends on the last estimated value (k). Moreover, to estimate $\hat{\Theta}_k$, we need to incorporate the current information present in $y_k$.

Aguirre BOOK defines the Recursive Least Squares estimator with forgetting factor $\lambda$ as

$$
\left\{\begin{array}{c}
K_k= Q_k\psi_k = \frac{P_{k-1} \psi_k}{\psi_k^{\mathrm{T}} P_{k-1} \psi_k+\lambda} \\
\hat{\theta}_k=\hat{\theta}_{k-1}+K_k\left[y(k)-\psi_k^{\mathrm{T}} \hat{\theta}_{k-1}\right] \\
P_k=\frac{1}{\lambda}\left(P_{k-1}-\frac{P_{k-1} \psi_k \psi_k^{\mathrm{T}} P_{k-1}}{\psi_k^{\mathrm{T}} P_{k-1} \psi_k+\lambda}\right)
\end{array}\right.
\tag{3.20}
$$

where $K_k$ is the gain vector calculation (also known as Kalman gain), $P_k$ is the covariance matrix update, and $y_k - \mathbf{\Psi}_k^T \theta_{k-1}$ is the a priori estimation error. The forgetting factor $\lambda$ ($0 < \lambda \leq 1$) is usually defined between $0.94$ and $0.99$. If you set $\lambda = 1$ you will be using the traditional recursive algorithm. The equation above consider that the regressor vector $\psi(k-1)$ has been rewritten as $\psi_k$, since this vector is updated at iteration $k$ and contains information up to time instant $k-1$. We can  Initialize the parameter estimate $\theta_0$ as

$$ \theta_0 = \mathbf{0} \tag{3.21}$$

and Initialize the inverse of the covariance matrix $\mathbf{P}_0$ with a large value:

$$ \mathbf{P}_0 = \frac{\mathbf{I}}{\delta} \tag{3.22}$$

where $\delta$ is a small positive constant, and $\mathbf{I}$ is the identity matrix.

The forgetting factor $\lambda$ controls how quickly the algorithm forgets past data:
- $\lambda = 1$ means no forgetting, and all past data are equally weighted.
- $\lambda < 1$ means that when new data is available, all weights are multiplied by $\lambda$, which can be interpreted as the ratio between consecutive weights for the same data.

You can access the source code to check how SysIdentPy implements the RLS algorithm. The following example present how you can use it in SysIdentPy.

```python
from sysidentpy.utils.generate_data import get_siso_data
from sysidentpy.basis_function import Polynomial
from sysidentpy.parameter_estimation import RecursiveLeastSquares
from sysidentpy.utils import build_lagged_matrix
import matplotlib.pyplot as plt

x_train, x_test, y_train, y_test = get_siso_data(
    n = 1000, colored_noise = False, sigma = 0.001, train_percentage = 90
)

xlag = 2
ylag = 2
max_lag = 2
regressor_matrix = build_lagged_matrix(
    x=x_train, y=y_train, xlag=xlag, ylag=ylag, model_type="NARMAX",
)
# apply the basis function
psi = Polynomial(degree=2).fit(regressor_matrix, max_lag=max_lag, xlag=xlag, ylag=ylag)
estimator = RecursiveLeastSquares(lam=0.99)
theta = estimator.optimize(psi, y_train[max_lag:, :])
theta

[[-1.1778e-04]
 [ 1.9988e-01]
 [-9.3114e-05]
 [ 2.5119e-04]
 [ 9.0006e-01]
 [ 1.8339e-04]
 [-1.1943e-04]
 [ 9.9957e-02]
 [-4.6181e-05]
 [ 1.3155e-04]
 [ 3.4535e-04]
 [ 1.3843e-04]
 [-3.5454e-05]
 [ 1.5669e-04]
 [ 2.4311e-04]]
```

You can plot the evolution of the estimated parameters over time by accessing the `theta_evolution` values
```python
# plotting only the first 50 values
plt.plot(estimator.theta_evolution.T[:50, :])
plt.xlabel("iterations")
plt.ylabel("theta")
```
![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/theta_evolution.png?raw=true)
> Figure 1. Evolution of the estimated parameters over time using the RLS algorithm.

## Least Mean Squares

The Least Mean Squares (LMS) adaptive filter is a popular stochastic gradient algorithm developed by Widrow and Hoff in 1960. The LMS adaptive filter aims to adaptively change its filter coefficients to achieve the best possible filtering of a signal. This is done by minimizing the error between the desired signal $d(n)$ and the filter output $y(n)$. We can derive the LMS algorithm from the RLS formulation.

In RLS, the $\lambda$ is related to the minimization of the sum of weighted squares of the innovation

$$
J_k = \sum^k_{j=1}\lambda^{k-j}e^2_j.
\tag{3.23}
$$

The $Q_k$ in Equation 3.20, defined as

$$
Q_k = \frac{P_{k-1}}{\psi_k^{\mathrm{T}} P_{k-1} \psi_k+\lambda} \\
\tag{3.24}
$$

is derived from the general form the Kalman Filter (KF) algorithm.

$$
Q_k = \frac{P_{k-1}}{\psi_k^{\mathrm{T}} P_{k-1} \psi_k+v_0} \\
\tag{3.25}
$$

where $v_0$ is the variance of the noise in the definition of the KF, in which the cost function is defined as the sum of squares of the innovation (noise). You can check the details in the [Billings, S. A. - Nonlinear System Identification: NARMAX Methods in the Time, Frequency, and Spatio-Temporal Domains](https://www.wiley.com/en-us/Nonlinear+System+Identification%3A+NARMAX+Methods+in+the+Time%2C+Frequency%2C+and+Spatio-Temporal+Domains-p-9781119943594).

If we change $Q_k$ in Equation 3.25 to scaled identity matrix

$$
Q_k = \frac{\mu}{\Vert \psi_k \Vert^2}I
\tag{3.26}
$$

where $\mu \in \mathbb{R}^+$, the $Q_k$ and $\hat{\theta}_k$ in Equation 3.20 becomes

$$
\hat{\theta}_k=\hat{\theta}_{k-1}+\frac{\mu\left[y(k)-\psi_k^{\mathrm{T}} \hat{\theta}_{k-1}\right]}{\Vert \psi_k \Vert^2}\psi_k
\tag{3.27}
$$

where $\psi_k^{\mathrm{T}} \hat{\theta}_{k-1} = \hat{y}_k$, which is known as the LMS algorithm.

#### Convergence and Step-Size

The step-size parameter $\mu$ plays a crucial role in the performance of the LMS algorithm. If $\mu$ is too large, the algorithm may become unstable and fail to converge. If $\mu$ is too small, the algorithm will converge slowly. The choice of $\mu$ is typically:

$$
0 < \mu < \frac{2}{\lambda_{\max}}
\tag{3.28}
$$

where $\lambda_{\max}$ is the largest eigenvalue of the input signal’s autocorrelation matrix.

In SysIdentPy, you can use several variants of the LMS algorithm:

1. **LeastMeanSquareMixedNorm**
2. **LeastMeanSquares**
3. **LeastMeanSquaresFourth**
4. **LeastMeanSquaresLeaky**
5. **LeastMeanSquaresNormalizedLeaky**
6. **LeastMeanSquaresNormalizedSignRegressor**
7. **LeastMeanSquaresNormalizedSignSign**
8. **LeastMeanSquaresSignError**
9. **LeastMeanSquaresSignSign**
10. **AffineLeastMeanSquares**
11. **NormalizedLeastMeanSquares**
12. **NormalizedLeastMeanSquaresSignError**
13. **LeastMeanSquaresSignRegressor**

To use any one on the methods above, you just need to import it and set the `estimator` using the option you want:

```python
from sysidentpy.utils.generate_data import get_siso_data
from sysidentpy.basis_function import Polynomial
from sysidentpy.parameter_estimation import LeastMeanSquares
from sysidentpy.utils import build_lagged_matrix

x_train, x_test, y_train, y_test = get_siso_data(
    n = 1000, colored_noise = False, sigma = 0.001, train_percentage = 90
)

xlag = 2
ylag = 2
max_lag = 2
regressor_matrix = build_lagged_matrix(
    x=x_train, y=y_train, xlag=xlag, ylag=ylag, model_type="NARMAX",
)
# apply the basis function
psi = Polynomial(degree=2).fit(regressor_matrix, max_lag=max_lag, xlag=xlag, ylag=ylag)
estimator = LeastMeanSquares(mu=0.1)
theta = estimator.optimize(psi, y_train[max_lag:, :])
theta

[[ 1.5924e-04]
 [ 1.9950e-01]
 [ 3.2137e-04]
 [ 1.7824e-04]
 [ 8.9951e-01]
 [ 2.7314e-04]
 [ 3.3538e-04]
 [ 1.0062e-01]
 [ 3.5219e-04]
 [ 1.3544e-04]
 [ 3.4149e-04]
 [ 5.6315e-04]
 [-4.6664e-04]
 [ 2.2849e-04]
 [ 1.0536e-04]]
```

## Extended Least Squares Algorithm

Let's show an example of the effect of a biased parameter estimation. To make things simple,The data is generated by simulating the following model:

$$
y_k = 0.2y_{k-1} + 0.1y_{k-1}x_{k-1} + 0.9x_{k-2} + e_{k}
$$

In this case, we know the values of the true parameters, so it will be easier to understand how they are affected by a biased estimation. The data is generated using a method from SysIdentPy. If *colored_noise* is set to True in the method, a colored noise is added to the data:

$$e_{k} = 0.8\nu_{k-1} + \nu_{k}$$

where $x$ is a uniformly distributed random variable and $\nu$ is a gaussian distributed variable with $\mu=0$ and $\sigma$ is defined by the user.

We will generate a data with 1000 samples with white noise and selecting 90% of the data to train the model.

```python
x_train, x_valid, y_train, y_valid = get_siso_data(
    n=1000, colored_noise=True, sigma=0.2, train_percentage=90
)
```

First we will train a model without the Extended Least Squares Algorithm for comparison purpose.

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial
from sysidentpy.parameter_estimation import LeastSquares
from sysidentpy.utils.generate_data import get_siso_data
from sysidentpy.utils.display_results import results

basis_function = Polynomial(degree=2)
estimator = LeastSquares(unbiased=False)
model = FROLS(
    order_selection=False,
    n_terms=3,
    ylag=2,
    xlag=2,
    info_criteria="aic",
    estimator=estimator,
    basis_function=basis_function,
    err_tol=None,
)

model.fit(X=x_train, y=y_train)
yhat = model.predict(X=x_valid, y=y_valid)
rrse = root_relative_squared_error(y_valid, yhat)

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

| Regressors    | Parameters | ERR            |
| ------------- | ---------- | -------------- |
| x1(k-2)       | 9.0442E-01 | 7.55518391E-01 |
| y(k-1)        | 2.7405E-01 | 7.57565084E-02 |
| x1(k-1)y(k-1) | 9.8757E-02 | 3.12896171E-03 |

Clearly we have something wrong with the obtained model. The estimated parameters differs from the true one defined in the equation that generated the data. As we can observe above, the model structure is exact the same the one that generate the data. You can se that the ERR ordered the terms in the correct way. And this is an important note regarding the ERR algorithm: __it is very robust to colored noise!!__

That is a great feature! However, although the structure is correct, the model *parameters* are not correct! Here we have a biased estimation! For instance, the real parameter for $y_{k-1}$ is $0.2$, not $0.274$.

In this case, we are actually modeling using a NARX model, not a NARMAX. The MA part exists to allow an unbiased estimation of the parameters. To achieve a unbiased estimation of the parameters we have the Extend Least Squares algorithm.

Before applying the Extended Least Squares Algorithm we will run several NARX models to check how different the estimated parameters are from the real ones.

```python
parameters = np.zeros([3, 50])
for i in range(50):
    x_train, x_valid, y_train, y_valid = get_siso_data(
        n=3000, colored_noise=True, train_percentage=90
    )
    model.fit(X=x_train, y=y_train)
    parameters[:, i] = model.theta.flatten()

# Set the theme for seaborn (optional)
sns.set_theme()
plt.figure(figsize=(14, 4))
# Plot KDE for each parameter
sns.kdeplot(parameters.T[:, 0], label='Parameter 1')
sns.kdeplot(parameters.T[:, 1], label='Parameter 2')
sns.kdeplot(parameters.T[:, 2], label='Parameter 3')
# Plot vertical lines where the real values must lie
plt.axvline(x=0.1, color='k', linestyle='--', label='Real Value 0.1')
plt.axvline(x=0.2, color='k', linestyle='--', label='Real Value 0.2')
plt.axvline(x=0.9, color='k', linestyle='--', label='Real Value 0.9')
plt.xlabel('Parameter Value')
plt.ylabel('Density')
plt.title('Kernel Density Estimate of Parameters')
plt.legend()
plt.show()
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/biased_parameter.png?raw=true)
> Figure 2.: Kernel Density Estimates (KDEs) of the estimated parameters obtained from 50 NARX models realizations, each fitted to data with colored noise. The vertical dashed lines indicate the true parameter values used to generate the data. While the model structure is correctly identified, the estimated parameters are biased due to the omission of the Moving Average (MA) component, highlighting the need for the Extended Least Squares algorithm to achieve unbiased parameter estimation


As shown in figure above, we have a problem to estimate the parameter for $y_{k-1}$. Now we will use the Extended Least Squares Algorithm. In SysIdentPy, just set `unbiased=True` in the parameter estimation definition and the ELS algorithm will be applied.

```python
basis_function = Polynomial(degree=2)
estimator = LeastSquares(unbiased=True)
parameters = np.zeros([3, 50])
for i in range(50):
    x_train, x_valid, y_train, y_valid = get_siso_data(
        n=3000, colored_noise=True, train_percentage=90
    )
    model = FROLS(
        order_selection=False,
        n_terms=3,
        ylag=2,
        xlag=2,
        elag=2,
        info_criteria="aic",
        estimator=estimator,
        basis_function=basis_function,
    )

    model.fit(X=x_train, y=y_train)
    parameters[:, i] = model.theta.flatten()

plt.figure(figsize=(14, 4))
# Plot KDE for each parameter
sns.kdeplot(parameters.T[:, 0], label='Parameter 1')
sns.kdeplot(parameters.T[:, 1], label='Parameter 2')
sns.kdeplot(parameters.T[:, 2], label='Parameter 3')
# Plot vertical lines where the real values must lie
plt.axvline(x=0.1, color='k', linestyle='--', label='Real Value 0.1')
plt.axvline(x=0.2, color='k', linestyle='--', label='Real Value 0.2')
plt.axvline(x=0.9, color='k', linestyle='--', label='Real Value 0.9')
plt.xlabel('Parameter Value')
plt.ylabel('Density')
plt.title('Kernel Density Estimate of Parameters')
plt.legend()
plt.show()
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/unbiased_estimator.png?raw=true)
> Figure 3. Kernel Density Estimates (KDEs) of the estimated parameters obtained from 50 NARX models using the Extended Least Squares (ELS) algorithm with unbiased estimation. The vertical dashed lines indicate the true parameter values used to generate the data.

Unlike the previous biased estimation, these KDEs in Figure 3 show that the estimated parameters are now closely aligned with the true values, demonstrating the effectiveness of the ELS algorithm in achieving unbiased parameter estimation, even in the presence of colored noise.

> The Extended Least Squares algorithm is iterative by nature. In SysIdentPy, the default number of iterations is set to 30 (`uiter=30`). However, the [literature](https://www.researchgate.net/publication/303679484_Introducao_a_Identificacao_de_Sistemas) suggests that the algorithm typically converges quickly, often within 10 to 20 iterations. Therefore, you may want to test different numbers of iterations to find the optimal balance between convergence speed and computational efficiency.

