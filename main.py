import numpy as np
from sys_identfy import sys_identfy
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")


decimacao = 12
file_loaded = np.loadtxt('buck_id.dat')
y = file_loaded[0::decimacao, 2]
u = file_loaded[0::decimacao, 1]
file_loaded = np.loadtxt('buck_val.dat')
y_valid_ = file_loaded[0::decimacao, 2]
u_valid_ = file_loaded[0::decimacao, 1]

u = np.reshape(u, (len(u), 1))
y = np.reshape(y, (len(y), 1))
y_valid_ = np.reshape(y_valid_, (len(y_valid_), 1))
u_valid_ = np.reshape(u_valid_, (len(u_valid_), 1))


# Exemplo 1 - Com valores padrão
model = sys_identfy(scoring='mean_squared_root')
# não precisa colocar nada, apenas model.fit(u, y).
# Só coloquei para printar o modelo
regressores, _, _, _ = model.fit(u, y)
y_test3 = model.predict(u_valid_, y_valid_)
rrse = model.score(y_valid_, y_test3)
print('o modelo é:', regressores)
print('O rrse é', rrse)

plt.plot(y_valid_)
plt.plot(y_test3, 'r--')
plt.show()

# Exemplo 2 - definindo alguns valores diferentes
model1 = sys_identfy(info_criteria='fpe', scoring='root_mean_squared_error')
regressores, _, _, _ = model1.fit(u, y)
y_test = model1.predict(u_valid_, y_valid_)
rmse = model1.score(y_valid_, y_test3)
print('o modelo é: ', regressores)
print('o rmse é:', rmse)

plt.plot(y_valid_)
plt.plot(y_test3, 'r--')
plt.show()
