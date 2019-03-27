import numpy as np
from sys_identfy import *
import pandas as pd
import matplotlib.pyplot as plt

non_degree = 2
ylag = 2
ulag = 2
#model_length = 7 #Valor ignorado caso se utilize o critério de informação

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

u = np.reshape(u, (len(u), 1))
y = np.reshape(y, (len(y), 1))
y_valid_ = np.reshape(y_valid_, (len(y_valid_), 1))
u_valid_ = np.reshape(u_valid_, (len(u_valid_), 1))

model1 = sys_identfy(non_degree,ylag,ulag)
reg_code_ = model1.reg_code

w = model1.build_information_matrix(reg_code_, u, y)

reference = np.arange(1,len(model1.reg_code)+1)
aic = model1.information_criterion(y,u,0)
plt.plot(reference, aic, 'o--')
plt.show()

# bic = model1.information_criterion(y,u,1)
# plt.plot(reference, bic, 'o--')
# plt.show()

# fpe = model1.information_criterion(y,u,2)
# plt.plot(reference, fpe, 'o--')
# plt.show()

# lilc = model1.information_criterion(y,u,3)
# plt.plot(reference, lilc, 'o--')
# plt.show()
model_length = input('Number of model elements:')
model_length = int(model_length)



[model, errr, pivv, psi] = model1.ERR(y,w,model_length)
number_of_elements, nno, maximum_lag, number_of_output, nu, new_model = model1.model_information(model)
theta = model1.last_squares(psi, y)

print('taxa de redução de erro é', errr)

print('o pivot é',pivv)

#Testando o método de simulação livre
y_test3 = model1.model_prediction(model,pivv,y_valid_,u_valid_,theta)

plt.plot(y_valid_)
plt.plot(y_test3,'r--')
plt.show()

""" mean_forecast = model1.mean_forecast_error(y_valid_, y_test3)
print('mean forecast', mean_forecast)
msle = model1.mean_squared_log_error(y_valid_, y_test3)
print('msle', msle)
mse1 = model1.mean_squared_error(y_valid_, y_test3)
print('mse1', mse1)
rmse1 = model1.root_mean_squared_error(y_valid_, y_test3)
print('rmse1', rmse1)
nrmse = model1.normalized_root_mean_squared_error(y_valid_, y_test3)
print('nrmse', nrmse)
expl = model1.explained_variance_score(y_valid_, y_test3)
print('expl', expl)
rrse = model1.root_relative_squared_error(y_valid_, y_test3)
print('rrse', rrse)
mae = model1.mean_absolute_error(y_valid_, y_test3)
print('mae', mae)
med_ae = model1.median_absolute_error(y_valid_, y_test3)
print('med_ae', med_ae)
#r2 = model1.r2_score(y_valid_, y_test3)
#print('r2', r2)
s_mape = model1.symmetric_mean_absolute_percentage_error(y_valid_, y_test3)
print('s_mape', s_mape) """

rrse = model1.root_relative_squared_error(y_valid_, y_test3)
print('rrse', rrse)
print('fim')
