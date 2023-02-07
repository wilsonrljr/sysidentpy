# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 21:33:02 2023

@author: Gabriel
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function._basis_function import Polynomial
from sysidentpy.multiobjective.im import IM
from sysidentpy.metrics import root_relative_squared_error
from sysidentpy.utils.display_results import results
from sysidentpy.utils.plotting import plot_residues_correlation, plot_results
from sysidentpy.residues.residues_correlation import compute_residues_autocorrelation, compute_cross_correlation
from sysidentpy.utils.generate_data import get_miso_data

x_train, x_valid, y_train, y_valid = get_miso_data(
    n=1000,
    colored_noise=False,
    sigma=0.05,
    train_percentage=90
    )

basis_function = Polynomial(degree=3)

model = FROLS(
    order_selection=True,
    n_info_values=5,
    n_terms=4,
    extended_least_squares=False,
    ylag=2, xlag=[[1, 2], [1, 2]],
    info_criteria='aic',
    estimator='least_squares',
    basis_function=basis_function
)

x=np.arange(0, 1, 0.01)
y=np.arange(0, 1, 0.01)
Y=(x**2+ y).reshape(-1,1)

U = np.zeros((len(x), 2))
U[:, 0] = x
U[:, 1] = y

#model.fit(X=x_train, y=y_train)
model.fit(X=x_train, y=y_train)

yhat = model.predict(X=x_valid, y=y_valid)

s = IM
w1, w2, ed, ee = ((s.PA(model, U, Y, y_train, 30,  x_valid, y_valid)))

result = {'Peso Dinâmico': w1,
          'Peso Estático': w2,
          'Erro Dinâmico': ed,
          'Erro Estático': ee}
print(pd.DataFrame(result))
