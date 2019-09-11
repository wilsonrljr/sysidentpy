#%%
import numpy as np
from sys_identfy import sys_identfy
import matplotlib.pyplot as plt
import pandas as pd

#Carrega os dados
decimation = 1
# file_loaded = np.loadtxt('sistema_simulado_id.csv')
file_loaded = np.array(np.genfromtxt('sistema_simulado_id.csv', dtype=float, delimiter=',', names=True).tolist())
y = file_loaded[0::decimation, 1]
u = file_loaded[0::decimation, 0]
file_loaded = np.array(np.genfromtxt('sistema_simulado_val.csv', dtype=float, delimiter=',', names=True).tolist())
y_valid_ = file_loaded[0::decimation, 1]
u_valid_ = file_loaded[0::decimation, 0]

u = np.reshape(u, (len(u), 1))
y = np.reshape(y, (len(y), 1))
y_valid_ = np.reshape(y_valid_, (len(y_valid_), 1))
u_valid_ = np.reshape(u_valid_, (len(u_valid_), 1))

# Chama a classe
model = sys_identfy(n_terms=3, non_degree=3, ylag=3, xlag=3, info_criteria='bic')

# Carrega os dados
model.fit(u, y)

# Toma o vetor simulado pelo modelo
y_test = model.predict(u_valid_, y_valid_)

# Calcula o índice de validação
rrse = model.score(y_valid_, y_test)

# Gera os resultados
results = pd.DataFrame(model.results(), columns=['Regressors', 'Parameters', 'ERR'])

#Exibe os resultados
print('RRSE:', rrse)
print(results)

# Plota os gráficos
import seaborn as sns
sns.set(style='whitegrid')
fig, ax = plt.subplots()
ax.plot(y_valid_, 'b-', label='Data', linewidth=5)
ax.plot(y_test, 'r--', label='Model', linewidth=5)
ax.legend(fontsize=32)
plt.xticks(fontsize=32)
plt.yticks(fontsize=32)
plt.xlabel('Samples', fontsize=32)
plt.ylabel('y', fontsize=32)
plt.show()

# Correlação
residuals = (y_valid_ - y_test).flatten()[2:]
r_residuals, ruy_residuals = model.residuals()

plt.plot(r_residuals[:,0], 'b-', linewidth=5)
plt.plot(r_residuals[:,1], 'k:', linewidth=5)
plt.plot(r_residuals[:,2], 'k:', linewidth=5)
plt.xticks(fontsize=32)
plt.yticks(fontsize=32)
plt.xlabel('Lag', fontsize=32)
plt.ylabel('Autocorrelation', fontsize=32)
axes = plt.gca()
axes.set_ylim([-1,1])
plt.show()

plt.plot(ruy_residuals[:,0], 'b-', linewidth=5)
plt.plot(ruy_residuals[:,1], 'k:', linewidth=5)
plt.plot(ruy_residuals[:,2], 'k:', linewidth=5)
plt.xticks(fontsize=32)
plt.yticks(fontsize=32)
plt.xlabel('Lag', fontsize=32)
plt.ylabel('Cross correlation', fontsize=32)
axes = plt.gca()
axes.set_ylim([-1,1])
plt.show()

#%%
