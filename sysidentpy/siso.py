import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function._basis_function import Polynomial
from sysidentpy.multiobjective.im import IM
from sysidentpy.metrics import root_relative_squared_error
from sysidentpy.utils.generate_data import get_siso_data
from sysidentpy.utils.display_results import results
from sysidentpy.utils.plotting import plot_residues_correlation, plot_results
from sysidentpy.residues.residues_correlation\
    import compute_residues_autocorrelation, compute_cross_correlation

x_train, x_valid, y_train, y_valid = get_siso_data(
    n=1000,
    colored_noise=True,
    sigma=0.05,
    train_percentage=90)

basis_function = Polynomial(degree=2)

model = FROLS(
    order_selection=False,
    n_terms=5,
    n_info_values=5,
    extended_least_squares=False,
    ylag=2, xlag=2,
    info_criteria='aic',
    estimator='least_squares',
    basis_function=basis_function
)
# Gerando a curva estática
U = (np.arange(1, 1.2, 0.001)).reshape(-1, 1)
Y = U**4 + 0.7*U
w = 1
model.fit(X=x_train, y=y_train)

yhat = model.predict(X=x_valid, y=y_valid)
# Chamando o método multiobjetivo para estimar os parâmetros
w1, w2, ed, es = ((IM.est_multi(model, U, Y, y_train, 30, x_valid, y_valid)))
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
point = 4
# Sendo mostrado os resultados gerado pelo ponto de operação
plot_results(y=y_valid, yhat=IM.plotim(model, point=point)[0], n=1000)
ee = compute_residues_autocorrelation(y_valid, IM.plotim(model,
                                      point=point)[0])
plot_residues_correlation(data=ee, title="Residues", ylabel="$e^2$")
x1e = compute_cross_correlation(y_valid, IM.plotim(model,
                                point=point)[0],
                                x_valid)
plot_residues_correlation(data=x1e, title="Residues", ylabel="$x_1e$")

print('Erro dinâmico: ', ed[point])

r = pd.DataFrame(
    results(
        model.final_model, model.theta, model.err,
        model.n_terms, err_precision=3, dtype='sci'
        ),
    columns=['Regressors', 'Parameters', 'ERR'])
print(r)

plt.figure(2)
plt.plot(U, IM.plotim(model, point=point)[1], 'b', linewidth=2, label='NARMAX')
plt.plot(U, Y, 'r', linewidth=2, label='Curva Estática')
plt.title('Modelo SISO - Curva Estática', fontsize=15)
plt.xlabel('U')
plt.ylabel('Y')
plt.legend()
print('Erro estático: ', es[point])