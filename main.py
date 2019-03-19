import numpy as np
from sys_identfy import *
import pandas as pd
import matplotlib.pyplot as plt


non_degree = 2
ylag = 2
ulag = 2
model_length = 4 #Valor ignorado caso se utilize o critério de informação

# porcent=70
# y_caminho='y1.txt'
# u_caminho='u1.txt'
# [y_ident_,u_ident_,y_valid_,u_valid_] = sys_identfy.prepare_data(y_caminho,u_caminho,porcent)
# y=y_ident_
# u=u_ident_

decimacao = 12
file_loaded = np.loadtxt('buck_id.dat')
y = file_loaded[0::decimacao,2]
u = file_loaded[0::decimacao,1]
file_loaded = np.loadtxt('buck_val.dat')
y_valid_ = file_loaded[0::decimacao,2]
u_valid_ = file_loaded[0::decimacao,1]


model1 = sys_identfy(non_degree,ylag,ulag)
reg_code_ = model1.reg_code
w = model1.get_regressmatrx(y,u)

reference = np.arange(1,len(model1.reg_code)+1)
# aic = model1.information_criterion(y,u,0)
# plt.plot(reference, aic, 'o--')
# plt.show()

# bic = model1.information_criterion(y,u,1)
# plt.plot(reference, bic, 'o--')
# plt.show()

# fpe = model1.information_criterion(y,u,2)
# plt.plot(reference, fpe, 'o--')
# plt.show()

lilc = model1.information_criterion(y,u,3)
plt.plot(reference, lilc, 'o--')
plt.show()
model_length = input('Number of model elements:')
model_length = int(model_length)


[model, errr, pivv, psi] = model1.ERR(y,w,model_length)
theta = model1.last_squares(psi,y)


#print(theta)
# print(pd.DataFrame(model))
print(errr)
# print(pivv)
# print(theta)

#Testando o método de simulação livre
y_test3 = model1.model_prediction(model,pivv,y_valid_,u_valid_,theta)

plt.plot(y_valid_)
plt.plot(y_test3,'r--')
plt.show()

[rmse, mse] = model1.validation_index(y_valid_,y_test3)

# print('rmse calculated: ', rmse)
# print('mse calculated: ', mse)

print('fim')
