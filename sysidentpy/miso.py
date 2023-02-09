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
# Gerando a curva estática
x = np.arange(0, 1, 0.01)
y = np.arange(0, 1, 0.01)
Y = (x**2 + y).reshape(-1, 1)
U = np.zeros((len(x), 2))
U[:, 0] = x
U[:, 1] = y

model.fit(X=x_train, y=y_train)
yhat = model.predict(X=x_valid, y=y_valid)
# Chamando o método multiobjetivo para estimar os parâmetros
w1, w2, ed, es = ((IM.est_multi(model, U, Y, y_train, 20,  x_valid, y_valid)))
# Mostrando os resultados
result = {'Peso Dinâmico': w1,
          'Peso Estático': w2,
          'Erro Dinâmico': ed,
          'Erro Estático': es}
print(pd.DataFrame(result))
# Plotando a cura de Pareto-Ótimo
plt.figure(1)
plt.title('Curva de Pareto-Ótimo', fontsize=15)
plt.plot(ed, es, 'xr', label='Pareto-Ótimo')
plt.xlabel('Objetivo 01 - Erro dinâmico')
plt.ylabel('Objetivo 02 - Erro estático')
plt.legend()
plt.show()
# Selecione o indíce de intesse
point = 2
# Sendo mostrado os resultados gerado pelo ponto de operação
plot_results(y=y_valid, yhat=IM.plotim(model,
                                       point=point)[0], n=1000)
ee = compute_residues_autocorrelation(y_valid, yhat)
plot_residues_correlation(data=ee, title="Residues", ylabel="$e^2$")
x1e = compute_cross_correlation(y_valid, IM.plotim(model,
                                point=point)[0], x_valid[:, 0])
plot_residues_correlation(data=x1e, title="Residues", ylabel="$x_1e$")
print('Erro dinâmico: ', ed[point])
r = pd.DataFrame(
    results(
        model.final_model, model.theta, model.err,
        model.n_terms, err_precision=3, dtype='sci'
        ),
    columns=['Regressors', 'Parameters', 'ERR'])
print(r)
x = U[:, 0]
y = U[:, 1]
z = x**2 + y
plt.figure(2)
ax = plt.axes(projection='3d')
ax.plot3D(x, y, IM.plotim(model,
                          point=point)[1][:, 0], 'green', linewidth=2,
                          label="Modelo NARMAX")
ax.plot3D(x, y, z, 'blue', linewidth=2, label="Curva Estática")
ax.set_title('Modelo MISO - Curva Estática', fontsize=15)
ax.set_xlabel('U1', fontsize=15)
ax.set_ylabel('U2', fontsize=15)
ax.set_zlabel('Y', fontsize=15)
print('Erro estático: ', es[point])
plt.legend()
plt.show() 