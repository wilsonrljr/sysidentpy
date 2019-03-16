import numpy as np
from sys_identfy import *
import pandas as pd
import matplotlib.pyplot as plt


non_degree = 2
ylag = 2
ulag = 2
ntp = 15
porcent=70
y_caminho='y1.txt'
u_caminho='u1.txt'

[y_ident_,u_ident_,y_valid_,u_valid_] = sys_identfy.prepare_data(y_caminho,u_caminho,porcent)


y=y_ident_
u=u_ident_


model1=sys_identfy(non_degree,ylag,ulag)
reg_code_=model1.reg_code
w=model1.get_regressmatrx(y,u)
[model, errr, pivv, psi] = model1.ERR(y,w,ntp)
theta=model1.last_squares(psi,y)


#print(theta)
print(pd.DataFrame(model))
# print(errr)
print(pivv)
# print(theta)

# y_control=psi@theta

#Esse método funcionou direito (mesmo resultado do y_control)
# w_temp1 = model1.get_regressmatrx(y,u) #usando a matriz código de regressores total, montou-se a matriz W
# pivv_temp = pivv[0:len(model)] #Tomou-se o vetor de reorganização de mesmo tamanho do modelo desejado (já já o Akaike "come solto, calma")
# w_temp1 = np.copy(w_temp1[:, pivv_temp]) #Reorganiza-se a matriz de regressores W de acordo com o vetor pivô retirando-se as colunas indesejadas
# y_test1=w_temp1@theta #Olha aí o sinal simulado do modelo...... Espero que funcione na função... desgraça!

#Esse método está dando muito errado (o sinal de saída apresenta módulo discrepante e misteriosamente com sinal invertido)
# w_temp2 = model1.get_regressmatrx(model,y,u) #lembrar de modificar esse método caso queira usar, mas não deu certo mesmo... então dane-se
# y_test2 = w_temp2@theta

#Testando o método de simulação livre
y_test3 = model1.model_prediction(model,pivv,y[0:2],u,theta)

# plt.plot(y)
# plt.plot(y_test3)
# plt.show()

#Corrigir, está dando valor estranho
# [rmse, mse] = model1.validation_index(y,y_test3)

# print(rmse)
# print(mse)



jaik = model1.akaike_information_criterion(y,u)

plt.plot(jaik)
plt.show()



#================================================

# y_livre = y[0:model1.max_lag]
# for i in range((model1.max_lag),u.size):
#     y_livre=np.append(y_livre,theta[0]*y_livre[i-1]+theta[1]*y_livre[i-2]+theta[2]*u[i-1]+theta[3]*y_livre[i-1]*u[i-1]+theta[4]*u[i-2]+theta[5]*y_livre[i-2]*u[i-1]+theta[6]*y_livre[i-1]*u[i-2]+theta[7]*y_livre[i-2]*u[i-2]+theta[8]*y_livre[i-1]*y_livre[i-1]+theta[9]*y_livre[i-2]*y_livre[i-2])
# print(y_livre)
# y_livre = np.zeros(u.size)
# y_livre[0:model1.max_lag] = np.copy(y[0:model1.max_lag])

# for i in range((model1.max_lag),u.size):
#     y_livre[i]=theta[0]*y_livre[i-1]+theta[1]*y_livre[i-2]+theta[2]*u[i-1]+theta[3]*y_livre[i-1]*u[i-1]+theta[4]*u[i-2]+theta[5]*y_livre[i-2]*u[i-1]+theta[6]*y_livre[i-1]*u[i-2]+theta[7]*y_livre[i-2]*u[i-2]+theta[8]*y_livre[i-1]*y_livre[i-1]+theta[9]*y_livre[i-2]*y_livre[i-2]
# print(y_livre)
# plt.plot(y)
# plt.plot(y_livre)
# plt.show()

print('fim')
