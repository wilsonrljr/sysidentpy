"""
Implementação da IM aplicada na estimação dos parâmetros
Autor: Gabriel Bueno Leandro
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.multiobjective.im import IM
from sysidentpy.basis_function._basis_function import Polynomial
from sysidentpy.utils.display_results import results
from sysidentpy.utils.plotting import plot_results
from sysidentpy.metrics import root_relative_squared_error
# Lendo os dados de entrada e saída do Buck
dados = pd.read_csv("buck.txt" , sep=" ", header=None).to_numpy()
dados_val = pd.read_csv("buck_val.txt" , sep =" ", header=None).to_numpy()

# Separando os dados de identificação
dados_amostras = dados[:, 2]
dados_saida = dados[:, 6]
dados_entrada = dados[:, 4]

# Separando os dados de validação
dados_amostras_val = dados_val[:, 2]
dados_saida_val = dados_val[:, 6]
dados_entrada_val = dados_val[:, 4]

# Plotando a saída medida(dados de identificação e validação)
plt.figure(1)
plt.title('Saída')
plt.plot(dados_amostras,dados_saida, 'b', label='Identificação', linewidth=3)
plt.plot(dados_amostras_val,dados_saida_val, 'r', label='Validação', linewidth=3)
plt.ylim(11, 17.7)
plt.legend()
plt.show()

# Plotando a entrada medida(dados de identificação e validação)
plt.figure(2)
plt.title('Entrada')
plt.plot(dados_amostras, dados_entrada, 'g', label='identificação',
         linewidth=3)
plt.plot(dados_amostras_val, dados_entrada_val, 'c', label='Validação',
         linewidth=3)
plt.legend()
plt.ylim(2.18, 2.6)
plt.show()

# Dados estáticos
Vd = 24
Yo = np.zeros(30)
Uo = np.zeros(30)
for t in range(10, 40):
    Uo[t-10] = (t)/10
    Yo[t-10] = (4-Uo[t-10])*Vd/3
Uo = Uo.reshape(-1, 1)
Yo = Yo.reshape(-1, 1)
plt.figure(3)
plt.title('Curva Estática Conversor Estático')
plt.xlabel('u')
plt.ylabel('y')
plt.plot(Uo, Yo, 'k', linewidth=3)
plt.show()

basis_function = Polynomial(degree=2)

model = FROLS(
    order_selection=True,
    n_info_values=10,
    extended_least_squares=False,
    ylag=2, xlag=2,
    info_criteria='aic',
    estimator='least_squares',
    basis_function=basis_function
)

# Informando os dados de teste e validação.
x_train = dados_entrada.reshape(-1, 1)
y_train = dados_saida.reshape(-1, 1)
x_valid = dados_entrada_val.reshape(-1, 1)
y_valid = dados_saida_val.reshape(-1, 1)

# O método fit executa o algoritmo de Taxa de Redução de Erros usando\
# a reflexão househoulder para selecionar a estrutura do modelo.
model.fit(X=x_train, y=y_train)

# O método de previsão é usado para gerar as previsões infinitos\
# passos a frente.
yhat = model.predict(X=x_valid, y=y_valid)

# Definindo o ganho
gain = Yo/24
plt.figure(3)
plt.title('Ganho do Conversor Estático')
plt.xlabel('$\\bar{u}$')
plt.ylabel('$\\bar{gain}$')
plt.plot(Uo, gain, 'c', linewidth=3)
plt.show()

# Matriz com os pesos
W = np.array([[0.98, 0.7, 0.5, 0.35, 0.25, 0.01, 0.15, 0.01],
              [0.01, 0.1, 0.3, 0.15, 0.25, 0.98, 0.35, 0.01],
              [0.01, 0.2, 0.2, 0.50, 0.50, 0.01, 0.50, 0.98]])

# Chamando a classe referente ao cálculo dos parâmetros via técnica\
# multiobjetivo
g = IM(y_train=y_train[2:],
       Gain=gain,Y_static=Yo,
       X_static=Uo, PSI=model.psi,
       n_inputs=model._n_inputs,
       non_degree=model.non_degree,
       model_type=model.model_type,
       final_model = model.final_model,
       W=W)
# Método para calcular os parâmetros
J, w, E, Theta, HR, QR = g.multio()
result = {'w1': w[0,:],
          'w2': w[2,:],
          'w3': w[1,:],
          'J_ls': J[0,:],
          'J_sg': J[1,:],
          'J_sf': J[2,:],
          '||J||:': E}
print(pd.DataFrame(result))

# Escrevendo os resultados
model.theta = Theta[0, :].reshape(-1,1)
yhat = model.predict(X=x_valid, y=y_valid)
rrse = root_relative_squared_error(y_valid, yhat)
r = pd.DataFrame(
    results(
        model.final_model, model.theta, model.err,
        model.n_terms, err_precision=3, dtype='sci'
        ),
    columns=['Regressors', 'Parameters', 'ERR'])
print(r)
plot_results(y=y_valid, yhat=yhat, n=1000)
# Plotando os gráficos resultantes
plt.figure(4)
plt.title('Ganho')
plt.plot(Uo, gain, 'g', linewidth=3, label='Ganho do CC-Buck')
plt.plot(Uo, HR.dot(model.theta), 'r', linewidth=3, label='Ganho do modelo NARX')
plt.xlabel('$\\bar{u}$')
plt.ylabel('$\\bar{g}$')
plt.legend()
plt.show()

plt.figure(5)
plt.title('Curva Estática')
plt.plot(Uo, Yo, 'b', linewidth=3, label='Curva Estática')
plt.plot(Uo, QR.dot(model.theta), 'm', linewidth=3, label='NARX representação estática')
plt.xlabel('$\\bar{u}$')
plt.xlabel('$\\bar{y}$')
plt.legend()
plt.show()

plt.figure(6)
ax = plt.axes(projection='3d')
ax.plot3D( J[0,:],  J[1,:],  J[2,:], 'bo', linewidth=2)
ax.set_title('Curva de Pareto-ótimo', fontsize=15)
ax.set_xlabel('$J_{ls}$', fontsize=15)
ax.set_ylabel('$J_{sg}$', fontsize=15)
ax.set_zlabel('$J_{sf}$', fontsize=15)
plt.show()