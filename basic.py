import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial, Fourier
from sysidentpy.metrics import root_relative_squared_error
from sysidentpy.utils.generate_data import get_siso_data
from sysidentpy.utils.display_results import results
from sysidentpy.utils.plotting import plot_residues_correlation, plot_results
from sysidentpy.residues.residues_correlation import compute_residues_autocorrelation, compute_cross_correlation

x_train, x_valid, y_train, y_valid = get_siso_data(
    n=1000,
    colored_noise=False,
    sigma=0.0001,
    train_percentage=90
)
# basis_function = Polynomial(degree=2)
basis_function = Fourier(degree=1, n=10, p=-0.5*(np.pi**2)) # p=-0.2*(np.pi**2)
# basis_function = Gaussian(N=60)

model = FROLS(
    order_selection=True,
    # n_terms=10,
    n_info_values=60,
    extended_least_squares=False,
    ylag=2, xlag=2,
    info_criteria='bic',
    estimator='least_squares',
    basis_function=basis_function
)
model.fit(x_train, y_train)
yhat = model.predict(x_valid, y_valid, steps_ahead=10)
rrse = root_relative_squared_error(y_valid, yhat)
print(rrse)

r = pd.DataFrame(
    results(
        model.final_model, model.theta, model.err,
        model.n_terms, err_precision=8, dtype='sci'
        ),
    columns=['Regressors', 'Parameters', 'ERR'])
print(r)

plot_results(y=y_valid, yhat = yhat, n=1000)
ee = compute_residues_autocorrelation(y_valid, yhat)
plot_residues_correlation(data=ee, title="Residues", ylabel="$e^2$")
x1e = compute_cross_correlation(y_valid, yhat, x_valid)
plot_residues_correlation(data=x1e, title="Residues", ylabel="$x_1e$")