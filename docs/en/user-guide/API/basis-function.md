---
template: overrides/main.html
---

# Documentation for `Basis Functions`

## Controlling the bias term

Basis functions that generate a global constant term expose the `include_bias`
option. It is available in `Polynomial`, `Bilinear`, `Bernstein`, `Legendre`,
`Hermite`, `HermiteNormalized`, and `Laguerre`. The default is `True`, which
preserves the historical generated feature matrices. Set it to `False` to
exclude the constant from the generated regressors and their model-structure
codes.

Canonical structure codes for existing non-`Polynomial` basis functions were
also corrected to describe their actual feature-column order. Consequently,
`regressor_code` and `final_model` metadata from those bases can differ from
earlier releases even though their default feature matrices are unchanged.
Entropic Regression may also select a different structure where it previously
mistook the first feature column for an intercept.

`Fourier` deliberately does not expose this option because its feature
expansion does not generate a constant term. Its public constructor and
generated feature matrix remain unchanged.

`include_bias` controls the features created by SysIdentPy. When using `NARX`
with an external estimator, check whether that estimator adds an intercept of
its own. For example, set `fit_intercept=False` on estimators that support that
option when the complete model must remain intercept-free.

::: sysidentpy.basis_function._bernstein
      show_root_heading: false

::: sysidentpy.basis_function._bilinear
      show_root_heading: false

::: sysidentpy.basis_function._fourier
      show_root_heading: false

::: sysidentpy.basis_function._legendre
      show_root_heading: false

::: sysidentpy.basis_function._polynomial
      show_root_heading: false

::: sysidentpy.basis_function._hermite
      show_root_heading: false

::: sysidentpy.basis_function._hermite_normalized
      show_root_heading: false

::: sysidentpy.basis_function._laguerre
      show_root_heading: false
