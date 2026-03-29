"""Build NARMAX Models using UOFR algorithm.

Implementation Notes
--------------------
This module implements the Ultra-Orthogonal Forward Regression (UOFR) algorithm
as described in [1]. Some implementation decisions extend or interpret the
original paper to handle practical edge cases:

1. **Normalization of antisymmetric kernels (Eq. 20-21)**:
   The paper requires normalized test functions to satisfy both unit L2 norm
   (Eq. 20) and unit integral (Eq. 21). However, odd-order derivatives of
   symmetric functions (e.g., 1st and 3rd derivatives of B-splines) are
   antisymmetric and have zero integral by definition. This implementation
   applies L2 normalization and returns a separate area correction factor
   for cases where the integral is non-zero.

2. **Length weighting for modulated signals**:
   The convolution with 'valid' mode reduces the number of samples in
   modulated signals. To prevent derivative terms from being underweighted
   in the ULS criterion (Eq. 24), we apply a scaling factor of
   sqrt(N_original / N_modulated). This is not explicitly mentioned in the
   paper but ensures balanced contribution across all Sobolev orders.

3. **Convolution vs. correlation (Eq. 25)**:
   The paper defines modulation as a sum that corresponds to cross-correlation.
   We implement this using np.convolve with a flipped kernel, which is
   mathematically equivalent to the correlation defined in Eq. 25.

4. **B-spline derivatives**:
   We use closed-form expressions for the cubic B-spline and its derivatives
   up to order 3, consistent with Appendix A of the paper. The cubic B-spline
   has continuous derivatives only up to order 2, so the 3rd derivative is
   piecewise constant.

References
----------
.. [1] Guo, Y., Guo, L.Z., Billings, S.A., Wei, H.L. (2015).
   "Ultra-Orthogonal Forward Regression Algorithms for the Identification
   of Non-Linear Dynamic Systems". Neurocomputing.
   https://eprints.whiterose.ac.uk/107310/1/UOFR%20Algorithms%20R1.pdf
"""

# Authors:
#           Wilson Rocha Lacerda Junior <wilsonrljr@outlook.com>
# License: BSD 3 clause

from typing import Union, Tuple, Optional, Callable

import numpy as np

from sysidentpy.narmax_base import house, rowhouse

from ..basis_function import Fourier, Polynomial
from .ofr_base import OFRBase, get_info_criteria, _compute_err_slice

from ..parameter_estimation.estimators import (
    LeastSquares,
    RidgeRegression,
    RecursiveLeastSquares,
    TotalLeastSquares,
    LeastMeanSquareMixedNorm,
    LeastMeanSquares,
    LeastMeanSquaresFourth,
    LeastMeanSquaresLeaky,
    LeastMeanSquaresNormalizedLeaky,
    LeastMeanSquaresNormalizedSignRegressor,
    LeastMeanSquaresNormalizedSignSign,
    LeastMeanSquaresSignError,
    LeastMeanSquaresSignSign,
    AffineLeastMeanSquares,
    NormalizedLeastMeanSquares,
    NormalizedLeastMeanSquaresSignError,
    LeastMeanSquaresSignRegressor,
)

Estimators = Union[
    LeastSquares,
    RidgeRegression,
    RecursiveLeastSquares,
    TotalLeastSquares,
    LeastMeanSquareMixedNorm,
    LeastMeanSquares,
    LeastMeanSquaresFourth,
    LeastMeanSquaresLeaky,
    LeastMeanSquaresNormalizedLeaky,
    LeastMeanSquaresNormalizedSignRegressor,
    LeastMeanSquaresNormalizedSignSign,
    LeastMeanSquaresSignError,
    LeastMeanSquaresSignSign,
    AffineLeastMeanSquares,
    NormalizedLeastMeanSquares,
    NormalizedLeastMeanSquaresSignError,
    LeastMeanSquaresSignRegressor,
]


class UOFR(OFRBase):
    r"""Ultra Orthogonal Forward Regression algorithm.

    This class uses the UOFR algorithm ([1]) to build NARMAX models.
    The NARMAX model is described as:

    $$
        y_k= F[y_{k-1}, \dotsc, y_{k-n_y},x_{k-d}, x_{k-d-1},
        \dotsc, x_{k-d-n_x}, e_{k-1}, \dotsc, e_{k-n_e}] + e_k
    $$

    where $n_y\in \mathbb{N}^*$, $n_x \in \mathbb{N}$, $n_e \in \mathbb{N}$,
    are the maximum lags for the system output and input respectively;
    $x_k \in \mathbb{R}^{n_x}$ is the system input and $y_k \in \mathbb{R}^{n_y}$
    is the system output at discrete time $k \in \mathbb{N}^n$;
    $e_k \in \mathbb{R}^{n_e}$ stands for uncertainties and possible noise
    at discrete time $k$. In this case, $\mathcal{F}$ is some nonlinear function
    of the input and output regressors and $d$ is a time delay typically set to
    $d=1$.

    The UOFR algorithm extends the classic OFR by using the Ultra-Least Squares
    (ULS) criterion, which measures model fitness in the Sobolev space H^m
    instead of the standard L^2 space. This criterion considers not only the
    residuals but also the weak derivatives of the signals, providing a stricter
    measure of model quality.

    The ULS criterion is defined as (Eq. 24 in [1]):

    $$
        J_{ULS} = \left\| y - \sum_{i=1}^{k} \theta_i x_i \right\|_2^2 +
        \sum_{l=1}^{m} \left\| \overline{y}^l - \sum_{i=1}^{k} \theta_i
        \overline{x}_i^l \right\|_2^2
    $$

    where $\overline{y}^l$ and $\overline{x}_i^l$ are the signals modulated
    by the normalized l-th derivative of the test function.

    Parameters
    ----------
    ylag : int, default=2
        The maximum lag of the output.
    xlag : int, default=2
        The maximum lag of the input.
    elag : int, default=2
        The maximum lag of the residues regressors.
    order_selection: bool, default=False
        Whether to use information criteria for order selection.
    info_criteria : str, default="aic"
        The information criteria method to be used.
    n_terms : int, default=None
        The number of the model terms to be selected.
        Note that n_terms overwrite the information criteria
        values.
    n_info_values : int, default=10
        The number of iterations of the information
        criteria method.
    estimator : str, default="least_squares"
        The parameter estimation method.
    model_type: str, default="NARMAX"
        The user can choose "NARMAX", "NAR" and "NFIR" models
    eps : float, default=np.finfo(np.float64).eps
        Normalization factor of the normalized filters.
    alpha : float, default=np.finfo(np.float64).eps
        Regularization parameter used in ridge regression.
        Ridge regression parameter that regularizes the algorithm to prevent over
        fitting. If the input is a noisy signal, the ridge parameter is likely to be
        set close to the noise level, at least as a starting point.
        Entered through the self data structure.
    sobolev_order : int, default=2
        Number of weak derivatives included in the Ultra Least Squares (ULS)
        augmentation (m in the manuscript, Eq. 22). Set to zero to disable
        augmentation and use standard OFR. The paper recommends m=2 for most
        applications (using 1st and 2nd derivatives).
    test_support : int, default=11
        Number of discrete samples used to represent the modulating function and
        its derivatives (n_0 in Eq. 25). An odd number is recommended so the
        kernel is centered. Larger values provide smoother modulation but reduce
        the number of valid samples after convolution.
    modulating_function : {"bspline", "gaussian"} or callable, default="bspline"
        Choice of test function used to smooth the signals before differentiating.
        The paper uses cubic B-spline (Appendix A) as it has finite support and
        continuous derivatives up to order 2. Options:

        - "bspline": Cubic B-spline as defined in Appendix A of [1].
        - "gaussian": Gaussian-like function with configurable sigma.
        - callable: Custom function with signature f(t, order) -> np.ndarray,
          where t is the grid and order is the derivative order.

    gaussian_sigma : float, default=1.0
        Standard deviation used when `modulating_function="gaussian"`.
        Only used if modulating_function is "gaussian".

    Notes
    -----
    **Implementation decisions not explicitly covered in the paper:**

    1. **Antisymmetric kernel normalization**: Odd-order derivatives of symmetric
       test functions have zero integral, making Eq. 21 impossible to satisfy
       directly. We handle this by applying L2 normalization and tracking area
       correction separately.

    2. **Sample count balancing**: Convolution reduces sample count. We scale
       modulated signals by sqrt(N_original/N_modulated) to maintain balanced
       contribution to the ULS criterion.

    3. **Parameter estimation**: The paper states parameters should be estimated
       using least squares on the original problem (Step 8). This implementation
       uses the configured estimator on the selected model terms.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from sysidentpy.model_structure_selection import UOFR
    >>> from sysidentpy.basis_function import Polynomial
    >>> from sysidentpy.utils.display_results import results
    >>> from sysidentpy.metrics import root_relative_squared_error
    >>> from sysidentpy.utils.generate_data import get_siso_data
    >>> x_train, x_valid, y_train, y_valid = get_siso_data(n=1000,
    ...                                                    colored_noise=True,
    ...                                                    sigma=0.2,
    ...                                                    train_percentage=90)
    >>> basis_function = Polynomial(degree=2)
    >>> model = UOFR(basis_function=basis_function,
    ...              order_selection=True,
    ...              n_info_values=10,
    ...              extended_least_squares=False,
    ...              ylag=2,
    ...              xlag=2,
    ...              info_criteria='aic',
    ...              sobolev_order=2,  # Use 1st and 2nd derivatives
    ...              )
    >>> model.fit(x_train, y_train)
    >>> yhat = model.predict(x_valid, y_valid)
    >>> rrse = root_relative_squared_error(y_valid, yhat)
    >>> print(rrse)
    0.001993603325328823
    >>> r = pd.DataFrame(
    ...     results(
    ...         model.final_model, model.theta, model.err,
    ...         model.n_terms, err_precision=8, dtype='sci'
    ...         ),
    ...     columns=['Regressors', 'Parameters', 'ERR'])
    >>> print(r)
        Regressors Parameters         ERR
    0        x1(k-2)     0.9000       0.0
    1         y(k-1)     0.1999       0.0
    2  x1(k-1)y(k-1)     0.1000       0.0

    References
    ----------
    .. [1] Guo, Y., Guo, L.Z., Billings, S.A., Wei, H.L. (2015).
       "Ultra-Orthogonal Forward Regression Algorithms for the Identification
       of Non-Linear Dynamic Systems". Neurocomputing.
       https://eprints.whiterose.ac.uk/107310/1/UOFR%20Algorithms%20R1.pdf

    """

    def __init__(
        self,
        *,
        ylag: Union[int, list] = 2,
        xlag: Union[int, list] = 2,
        elag: Union[int, list] = 2,
        order_selection: bool = True,
        info_criteria: str = "aic",
        n_terms: Union[int, None] = None,
        n_info_values: int = 15,
        estimator: Estimators = RecursiveLeastSquares(),
        basis_function: Union[Polynomial, Fourier] = Polynomial(),
        model_type: str = "NARMAX",
        eps: np.float64 = np.finfo(np.float64).eps,
        alpha: float = 0,
        err_tol: Optional[float] = None,
        apress_lambda: float = 1.0,
        sobolev_order: int = 2,
        test_support: int = 11,
        modulating_function: Union[
            str, Callable[[np.ndarray, int], np.ndarray]
        ] = "bspline",
        gaussian_sigma: float = 1.0,
    ):
        self.order_selection = order_selection
        self.ylag = ylag
        self.xlag = xlag
        self.max_lag = self._get_max_lag()
        self.info_criteria = info_criteria
        self.apress_lambda = apress_lambda
        self.info_criteria_function = get_info_criteria(info_criteria, apress_lambda)
        self.n_info_values = n_info_values
        self.n_terms = n_terms
        self.estimator = estimator
        self.elag = elag
        self.model_type = model_type
        self.basis_function = basis_function
        self.eps = eps
        if isinstance(self.estimator, RidgeRegression):
            self.alpha = self.estimator.alpha
        else:
            self.alpha = alpha

        self.err_tol = err_tol
        self.sobolev_order = sobolev_order
        self.test_support = test_support
        self.modulating_function = modulating_function
        self.gaussian_sigma = gaussian_sigma
        self._validate_params()
        self.n_inputs = None
        self.regressor_code = None
        self.info_values = None
        self.err = None
        self.final_model = None
        self.theta = None
        self.pivv = None
        self._validate_uofr_params()

    def _validate_uofr_params(self) -> None:
        """Validate UOFR-specific parameters.

        Raises
        ------
        ValueError
            If any UOFR parameter is invalid.
        TypeError
            If modulating_function is neither a string nor callable.
        """
        if not isinstance(self.sobolev_order, int) or self.sobolev_order < 0:
            raise ValueError(
                f"sobolev_order must be an integer >= 0. Got {self.sobolev_order}"
            )

        if not isinstance(self.test_support, int) or self.test_support < 3:
            raise ValueError(
                f"test_support must be an integer >= 3. Got {self.test_support}"
            )

        if self.test_support % 2 == 0:
            raise ValueError("test_support must be odd to center the test function.")

        if isinstance(self.modulating_function, str):
            allowed = {"bspline", "gaussian"}
            if self.modulating_function not in allowed:
                raise ValueError(
                    "modulating_function must be 'bspline', 'gaussian', or a callable."
                )
            if self.modulating_function == "bspline" and self.sobolev_order > 3:
                raise ValueError(
                    "B-spline modulating function currently supports"
                    " derivatives up to order 3. Use a custom callable"
                    " if higher Sobolev orders are required."
                )
        elif not callable(self.modulating_function):
            raise TypeError(
                "modulating_function must be a callable"
                " or one of the supported strings."
            )

        if (
            not isinstance(self.gaussian_sigma, (int, float))
            or self.gaussian_sigma <= 0
        ):
            raise ValueError(
                f"gaussian_sigma must be a positive float. Got {self.gaussian_sigma}"
            )

    def _test_function_grid(self) -> np.ndarray:
        """Generate the discrete grid for evaluating the test function.

        The grid is centered at zero and spans the support of the test function.
        For B-splines, the support is [-2, 2] (cubic B-spline, Appendix A).
        For Gaussian, the support is [-3*sigma, 3*sigma].

        Returns
        -------
        np.ndarray
            1D array of grid points with length equal to test_support.

        Notes
        -----
        The paper (Appendix A) defines the B-spline support implicitly through
        the knot sequence. Our implementation uses a centered cubic B-spline
        with support on [-2, 2].
        """
        if isinstance(self.modulating_function, str):
            if self.modulating_function == "bspline":
                # Cubic B-spline has support [-2, 2] (Appendix A)
                span = 2.0
            else:
                span = 3.0 * float(self.gaussian_sigma)
        else:
            # Default span for custom functions
            span = 1.0
        return np.linspace(-span, span, self.test_support)

    def _evaluate_test_function(self, t: np.ndarray, order: int) -> np.ndarray:
        """Evaluate the test function or its derivatives at given points.

        Parameters
        ----------
        t : np.ndarray
            Grid points where to evaluate the function.
        order : int
            Derivative order (0 = function itself, 1 = first derivative, etc.)

        Returns
        -------
        np.ndarray
            Function values at the grid points.

        Notes
        -----
        This corresponds to computing phi^(l)(t) as described in Step 2 of the
        UOFR algorithm in Section 4.
        """
        if isinstance(self.modulating_function, str):
            if self.modulating_function == "bspline":
                return self._bspline_kernel(t, order)
            return self.gaussian_test_function(t, order)
        return self.modulating_function(t, order)

    def _bspline_kernel(self, t: np.ndarray, order: int) -> np.ndarray:
        """Evaluate cubic B-spline and derivatives up to order 3.

        Implements the cubic B-spline basis function as defined in Appendix A
        (Eq. 39-42) of the paper. The B-spline is centered at t=0 with support
        on [-2, 2].

        The cubic B-spline is defined as:
            B_3(t) = {
                (2/3) - |t|^2 + (1/2)|t|^3,  if |t| < 1
                (1/6)(2 - |t|)^3,             if 1 <= |t| < 2
                0,                            otherwise
            }

        Parameters
        ----------
        t : np.ndarray
            Grid points where to evaluate the function.
        order : int
            Derivative order (0, 1, 2, or 3).

        Returns
        -------
        np.ndarray
            B-spline or derivative values at the grid points.

        Notes
        -----
        - Order 0: The B-spline itself (symmetric, positive).
        - Order 1: First derivative (antisymmetric, zero integral).
        - Order 2: Second derivative (symmetric).
        - Order 3: Third derivative (piecewise constant, antisymmetric).

        The derivatives are computed analytically following Eq. 41-42, not
        numerically, to avoid discretization errors.

        The third derivative is included for completeness but note that the
        cubic B-spline only has continuous derivatives up to order 2.
        """
        abs_t = np.abs(t)
        sign_t = np.sign(t)

        if order == 0:
            # Cubic B-spline: B_3(t) - Eq. 39 in Appendix A
            result = np.zeros_like(t)
            mask_inner = abs_t < 1
            result[mask_inner] = (
                (2.0 / 3.0) - abs_t[mask_inner] ** 2 + 0.5 * abs_t[mask_inner] ** 3
            )
            mask_outer = (abs_t >= 1) & (abs_t < 2)
            result[mask_outer] = ((2 - abs_t[mask_outer]) ** 3) / 6.0
            return result

        if order == 1:
            # First derivative: d/dt B_3(t) - Eq. 41 in Appendix A
            # For |t| < 1: d/dt[(2/3) - t^2 + (1/2)|t|^3] = -2t + (3/2)t|t|
            # For 1 <= |t| < 2: d/dt[(1/6)(2-|t|)^3] = -(1/2)(2-|t|)^2 * sign(t)
            derivative = np.zeros_like(t)
            mask_inner = abs_t < 1
            derivative[mask_inner] = (
                -2 * abs_t[mask_inner] + 1.5 * abs_t[mask_inner] ** 2
            )
            mask_outer = (abs_t >= 1) & (abs_t < 2)
            derivative[mask_outer] = -0.5 * (2 - abs_t[mask_outer]) ** 2
            # Multiply by sign(t) because d|t|/dt = sign(t)
            return derivative * sign_t

        if order == 2:
            # Second derivative: d^2/dt^2 B_3(t) - Eq. 42 in Appendix A
            # For |t| < 1: -2 + 3|t|
            # For 1 <= |t| < 2: (2 - |t|)
            # Note: This is symmetric, so no sign multiplication needed
            second_derivative = np.zeros_like(t)
            mask_inner = abs_t < 1
            second_derivative[mask_inner] = -2 + 3 * abs_t[mask_inner]
            mask_outer = (abs_t >= 1) & (abs_t < 2)
            second_derivative[mask_outer] = 2 - abs_t[mask_outer]
            return second_derivative

        if order == 3:
            # Third derivative: d^3/dt^3 B_3(t)
            # Piecewise constant (B-spline has C^2 continuity)
            # For |t| < 1: 3 * sign(t)
            # For 1 <= |t| < 2: -sign(t)
            third_derivative = np.zeros_like(t)
            mask_inner = abs_t < 1
            third_derivative[mask_inner] = 3
            mask_outer = (abs_t >= 1) & (abs_t < 2)
            third_derivative[mask_outer] = -1
            return third_derivative * sign_t

        # For orders > 3, return zeros (B-spline derivatives vanish)
        return np.zeros_like(t)

    def gaussian_test_function(self, t: np.ndarray, order: int) -> np.ndarray:
        """Generate Gaussian-like test function and its derivatives.

        Provides an alternative to B-splines with infinite smoothness,
        though truncated to finite support for practical computation.

        Parameters
        ----------
        t : np.ndarray
            Grid points where to evaluate the function.
        order : int
            Derivative order (0 = function itself).

        Returns
        -------
        np.ndarray
            Gaussian or derivative values at the grid points.

        Notes
        -----
        Unlike B-splines, the Gaussian has infinite derivatives. However,
        derivatives are computed numerically using np.gradient, which may
        introduce discretization errors for high orders. For high-order
        Sobolev spaces, consider using analytic Hermite polynomial
        expressions or a custom callable.
        """
        sigma = float(self.gaussian_sigma)
        gaussian = np.exp(-(t**2) / (2 * sigma**2))

        # Truncate to finite support
        span = np.max(np.abs(t))
        gaussian[np.abs(t) >= span] = 0.0

        if order == 0:
            return gaussian

        derivative = gaussian.copy()
        for _ in range(order):
            derivative = np.gradient(derivative, t)
            derivative[np.abs(t) >= span] = 0.0

        return derivative

    def normalize_test_function(self, phi_j: np.ndarray) -> Tuple[np.ndarray, float]:
        """Normalize test function derivative following Eq. 20-21.

        The paper requires (Eq. 20):
            phi_bar^l = phi^(l) / ||phi^(l)||_2

        And (Eq. 21):
            integral(phi_bar^l) = 1

        However, for antisymmetric kernels (odd-order derivatives of symmetric
        functions), the integral is identically zero. This implementation
        returns the L2-normalized kernel and a separate area correction factor.

        Parameters
        ----------
        phi_j : np.ndarray
            The test function derivative to normalize.

        Returns
        -------
        kernel : np.ndarray
            L2-normalized kernel (satisfies Eq. 20).
        area_correction : float
            Factor to scale signals so effective integration equals 1.
            For antisymmetric kernels, returns 1.0 (no correction possible).

        Raises
        ------
        ValueError
            If the kernel has zero energy (all zeros).

        Notes
        -----
        **Implementation decision**: The paper's Eq. 21 cannot be satisfied
        for antisymmetric kernels. We interpret this as follows:

        - Apply L2 normalization (Eq. 20) to ensure balanced energy contribution.
        - For symmetric kernels with non-zero integral, compute area correction.
        - For antisymmetric kernels, use area_correction=1.0 and rely on the
          L2 normalization to balance the criterion.

        This interpretation ensures that each derivative order contributes
        proportionally to the ULS criterion without amplifying noise.
        """
        norm = np.linalg.norm(phi_j, ord=2)
        if norm == 0:
            raise ValueError("Cannot normalize a zero-energy modulating function.")

        kernel = phi_j / norm

        integral = kernel.sum()
        if np.isclose(integral, 0.0, atol=self.eps):
            # Antisymmetric kernel: integral is zero by symmetry
            # No area correction possible; rely on L2 normalization only
            area_correction = 1.0
        else:
            # Symmetric kernel: correct to unit integral
            area_correction = 1.0 / float(integral)

        return kernel, area_correction

    def compute_modulated_signal(
        self, signal: np.ndarray, phi_bar_j: np.ndarray
    ) -> np.ndarray:
        """Apply discrete convolution matching Eq. 25 of the paper.

        Computes the modulated signal:
            y_bar^l(k) = sum_{n=k}^{k+n_0} y(n) * phi_bar^l(n-k)

        This is equivalent to cross-correlation, implemented via convolution
        with a flipped kernel.

        Parameters
        ----------
        signal : np.ndarray
            The signal to modulate (y or x_i).
        phi_bar_j : np.ndarray
            The normalized modulating kernel.

        Returns
        -------
        np.ndarray
            Modulated signal with length N - n_0, where N is the original
            signal length and n_0 is the kernel support (test_support - 1).

        Raises
        ------
        ValueError
            If the kernel is empty or larger than the signal.

        Notes
        -----
        **Implementation decision**: We use np.convolve with mode='valid' and
        a flipped kernel. This is mathematically equivalent to the summation
        in Eq. 25:

            np.convolve(signal, kernel[::-1], 'valid')[k] =
            sum_{n=0}^{n_0} signal[k+n] * kernel[n]

        The 'valid' mode ensures we only compute output where the full kernel
        overlaps with the signal, avoiding boundary effects.
        """
        flattened_signal = np.asarray(signal, dtype=np.float64).reshape(-1)
        kernel = np.asarray(phi_bar_j, dtype=np.float64).reshape(-1)

        kernel_size = kernel.size
        if kernel_size == 0:
            raise ValueError("modulating kernel cannot be empty")
        if flattened_signal.size < kernel_size:
            raise ValueError(
                "test_support is too large for the available samples; "
                "decrease test_support or provide longer signals."
            )

        # Flip kernel to convert convolution to correlation (matching Eq. 25)
        reversed_kernel = kernel[::-1]
        modulated = np.convolve(flattened_signal, reversed_kernel, mode="valid")

        return modulated

    def augment_uls_terms(
        self, y: np.ndarray, psi: np.ndarray, m: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Augment the regression problem to form the ULS problem (Eq. 22-28).

        Constructs the augmented matrices Y_ULS and Phi_ULS by stacking the
        original signals with their modulated versions for each derivative order.

        Parameters
        ----------
        y : np.ndarray
            Output signal vector of shape (N,) or (N, 1).
        psi : np.ndarray
            Regressor matrix of shape (N, num_terms).
        m : int, optional
            Sobolev order (number of derivative levels). Defaults to
            self.sobolev_order.

        Returns
        -------
        y_augmented : np.ndarray
            Augmented output vector Y_ULS as in Eq. 27.
            Shape: (N + m*(N-n_0), 1)
        psi_augmented : np.ndarray
            Augmented regressor matrix Phi_ULS as in Eq. 28.
            Shape: (N + m*(N-n_0), num_terms)

        Notes
        -----
        **Implementation decisions**:

        1. **Length weighting**: The modulated signals have fewer samples
           (N - n_0) due to 'valid' convolution. To ensure balanced contribution
           to the ULS criterion, we scale by sqrt(N_original / N_modulated).
           This is not explicitly mentioned in the paper but prevents derivative
           terms from being underweighted.

        2. **Area correction**: Applied after L2 normalization to approximate
           the unit integral requirement (Eq. 21) for symmetric kernels.

        The augmented system has the form:
            [y; y_bar^1; ...; y_bar^m] = [psi; psi_bar^1; ...; psi_bar^m] * theta

        where the semicolon denotes vertical stacking.
        """
        y = y.reshape(-1, 1)
        if m is None:
            m = self.sobolev_order

        if m == 0:
            # No augmentation: standard least squares
            return y, psi

        base_length = y.shape[0]
        num_terms = psi.shape[1]
        y_augmented = y.copy()
        psi_augmented = psi.copy()
        t = self._test_function_grid()

        for j in range(1, m + 1):
            phi_j = self._evaluate_test_function(t, order=j)
            phi_bar_j, area_correction = self.normalize_test_function(phi_j)
            y_j = self.compute_modulated_signal(y[:, 0], phi_bar_j).reshape(-1, 1)
            modulated_length = y_j.shape[0]

            # **Implementation decision**: Length weighting
            # The paper's Eq. 24 sums terms directly, but modulated signals
            # have fewer samples. We scale to balance contribution:
            #   weight = sqrt(N / N_modulated)
            # This ensures that ||y_bar^l||^2 in the criterion is comparable
            # to ||y||^2 in terms of sample count influence.
            length_weight = np.sqrt(base_length / modulated_length)
            modulation_scale = area_correction * length_weight
            y_j *= modulation_scale

            y_augmented = np.vstack([y_augmented, y_j])

            modulated_terms = np.zeros((modulated_length, num_terms))
            for term in range(num_terms):
                modulated_terms[:, term] = self.compute_modulated_signal(
                    psi[:, term], phi_bar_j
                )

            modulated_terms *= modulation_scale
            psi_augmented = np.vstack([psi_augmented, modulated_terms])

        return y_augmented, psi_augmented

    def sobolev_error_reduction_ratio(
        self,
        psi: np.ndarray,
        y: np.ndarray,
        process_term_number: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute ERR on the ULS-augmented regression problem.

        Implements Steps 5-7 of the UOFR algorithm: compute ERR significance
        for each term using the augmented ULS matrices and select terms in
        a forward greedy manner.

        The ERR is computed as (Eq. 33 in the paper):
            ERR(phi_k) = <w_k, y>^2 / (<w_k, w_k> * <y, y>)

        where w_k is the orthogonalized regressor.

        Parameters
        ----------
        psi : np.ndarray
            Original regressor matrix of shape (N, num_terms).
        y : np.ndarray
            Output signal of shape (N, 1).
        process_term_number : int
            Maximum number of terms to select.

        Returns
        -------
        err : np.ndarray
            ERR values for each selected term (Eq. 33 computed on ULS problem).
        piv : np.ndarray
            Indices of selected terms in order of selection.
        psi_orthogonal : np.ndarray
            Augmented regressor matrix with selected columns.
        y_augmented : np.ndarray
            Augmented output vector.

        Notes
        -----
        The ERR is computed using the augmented matrices (Y_ULS, Phi_ULS),
        which means term significance is evaluated considering both the
        original fit and the derivative fits. This is the key difference
        from standard OFR: terms that appear significant under L2 may be
        less significant under the Sobolev norm, and vice versa.

        The orthogonalization uses Householder reflections for numerical
        stability, as mentioned in the paper (any orthogonalization method
        is valid, but Householder is preferred for large problems).
        """
        y_target = y[self.max_lag :, 0].reshape(-1, 1)
        y_augmented, psi_augmented = self.augment_uls_terms(
            y_target, psi, self.sobolev_order
        )
        y_augmented = y_augmented.reshape(-1, 1)
        squared_y = np.dot(y_augmented.T, y_augmented)
        squared_y = float(np.maximum(squared_y, np.finfo(np.float64).eps))
        psi_working = psi_augmented.copy()
        y_working = y_augmented.copy()
        num_terms = psi_working.shape[1]
        piv = np.arange(num_terms)
        candidate_err = np.zeros(num_terms)
        err = np.zeros(num_terms)

        for step_idx in np.arange(0, num_terms):
            candidate_err[step_idx:] = _compute_err_slice(
                psi_working,
                y_working,
                step_idx,
                squared_y,
                self.alpha,
                self.eps,
            )

            max_err_idx = np.argmax(candidate_err[step_idx:]) + step_idx
            err[step_idx] = candidate_err[max_err_idx]

            if step_idx == process_term_number:
                break

            if (self.err_tol is not None) and (err.cumsum()[step_idx] >= self.err_tol):
                self.n_terms = step_idx + 1
                process_term_number = step_idx + 1
                break

            psi_working[:, [max_err_idx, step_idx]] = psi_working[
                :, [step_idx, max_err_idx]
            ]
            piv[[max_err_idx, step_idx]] = piv[[step_idx, max_err_idx]]

            reflector = house(psi_working[step_idx:, step_idx])
            row_result = rowhouse(psi_working[step_idx:, step_idx:], reflector)
            y_working[step_idx:] = rowhouse(y_working[step_idx:], reflector)
            psi_working[step_idx:, step_idx:] = np.copy(row_result)

        tmp_piv = piv[0:process_term_number]
        psi_orthogonal = psi_augmented[:, tmp_piv]

        return err, tmp_piv, psi_orthogonal, y_augmented

    def run_mss_algorithm(
        self, psi: np.ndarray, y: np.ndarray, process_term_number: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run the model structure selection algorithm.

        This method overrides the base class to use the UOFR algorithm
        instead of standard OFR.

        Parameters
        ----------
        psi : np.ndarray
            Regressor matrix.
        y : np.ndarray
            Output signal.
        process_term_number : int
            Maximum number of terms to select.

        Returns
        -------
        tuple
            (err, piv, psi_selected, y_target) as returned by
            sobolev_error_reduction_ratio.
        """
        return self.sobolev_error_reduction_ratio(psi, y, process_term_number)

    def fit(self, *, X: Optional[np.ndarray] = None, y: np.ndarray):
        """Fit polynomial NARMAX model.

        This is an 'alpha' version of the 'fit' function which allows
        a friendly usage by the user. Given two arguments, x and y, fit
        training data.

        Parameters
        ----------
        X : ndarray of floats
            The input data to be used in the training process.
        y : ndarray of floats
            The output data to be used in the training process.

        Returns
        -------
        model : ndarray of int
            The model code representation.
        piv : array-like of shape = number_of_model_elements
            Contains the index to put the regressors in the correct order
            based on err values.
        theta : array-like of shape = number_of_model_elements
            The estimated parameters of the model.
        err : array-like of shape = number_of_model_elements
            The respective ERR calculated for each regressor.
        info_values : array-like of shape = n_regressor
            Vector with values of akaike's information criterion
            for models with N terms (where N is the
            vector position + 1).

        """
        super().fit(X=X, y=y)
        return self

    def predict(
        self,
        *,
        X: Optional[np.ndarray] = None,
        y: np.ndarray,
        steps_ahead: Optional[int] = None,
        forecast_horizon: Optional[int] = None,
    ) -> np.ndarray:
        """Predict output using the fitted UOFR model.

        Parameters
        ----------
        X : ndarray of floats, optional
            Input data for prediction.
        y : ndarray of floats
            Output data (initial conditions for simulation).
        steps_ahead : int, optional
            Number of steps ahead for prediction.
        forecast_horizon : int, optional
            Forecast horizon for multi-step prediction.

        Returns
        -------
        yhat : np.ndarray
            Predicted output values.
        """
        yhat = super().predict(
            X=X, y=y, steps_ahead=steps_ahead, forecast_horizon=forecast_horizon
        )
        return yhat
