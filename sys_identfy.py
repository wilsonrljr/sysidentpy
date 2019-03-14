#-*- coding: utf-8 -*-
# Copyright 2019 The SysIdentfy. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#==================================================================================================================================================================



class sys_identfy:
    import numpy as np
    import pandas as pd
    global get_regresvec
    global genreg
    # global get_regresvec
    global max_lag


    """
    This funcrion create a new object of the type sys_identfy. It is called automatically always that a new
    object is instantiated. It is responsible for create the variables from the object and call the function
    genreg that handle the hard job.
        Args:
            non_degree = Nonlinearity degree
            ylag = Max lag of input
            ulag = Max lag of output

    """
    def __init__(self,non_degree_,ylag_,ulag_):
        self.non_degree=non_degree_ # non_degree stores the nonlinearity degree
        self.ylag=ylag_             # ylag stores the maximum output lag
        self.ulag=ulag_             # ulag ylag stores the maximum input lag
        [self.reg_code,self.max_lag]=genreg(non_degree_,ylag_,ulag_) # reg_code stores all possible combinations
    #=============================================================================================================================================================


    """
    This function generates a codification from all possibles regressors given the maximum lag of the
    input and output.

        Args:
            non_degree = Nonlinearity degree
            ylag = Max lag of input
            ulag = Max lag of output
        Raises:
            max_lag = The max lag (to be used by another funcs)
            reg_code = Matrix with contains a codification from all possible
            regressors like the exemplo below
                Example:
                100n = y(k-n)
                200n = u(k-n)
                [100n 100n] = y(k-n)y(k-n)
                [200n 200n] = u(k-n)u(k-n)
    """
    def genreg(non_degree, ylag, ulag):
        # assert type(non_degree)==int and type(ylag)==int and type(ulag)==int,"Non_degree, ylag and ulag must be integer"
        from itertools import combinations_with_replacement
        import numpy as np
        # regressor's code
        ylagmax = 1001 + ylag
        ulagmax = 2001 + ulag
        y_vec = np.arange(1001, ylagmax)
        u_vec = np.arange(2001, ulagmax)
        r = np.array([0])
        r = np.concatenate([r, y_vec, u_vec])
        # generate the matrix with all possible regressors and retur the number of cols(a) and rols(b)
        reg_code = list(combinations_with_replacement(r, non_degree))
        reg_code = np.array(reg_code)
        [a, b] = reg_code.shape
        max_lag = max(ylag, ulag)
        return reg_code, max_lag
    #=============================================================================================================================================================

    """
    This function extrac a givn regressor from a dataset
        Args:
            reg_code = A ndarray that contains the mappin of all possible regressors (it must be generate with genreg() func. )
            i = Roll from the map
            j = Col from the map
            y = Vec that contains the outputs
            u = Vec that contains the inputs
        Raises:
            vec= A vector that contain the refer regressor
    """
    def get_regresvec(self,reg_code_,i,j,y,u):
        import numpy as np
        max_lag=self.max_lag
        aux_1=reg_code_-1000
        aux_2=reg_code_-2000
        if reg_code_[i,j] == 0:
            vec = np.ones(len(y)-(self.max_lag))
        if aux_1[i,j] > 0 and aux_1[i,j] < 100:
            pra=int(max_lag-aux_1[i,j])
            prb=int(len(y)-aux_1[i,j])
            vec = y[pra:prb]
        if aux_2[i,j] > 0 and aux_2[i,j] < 100:
            pra=int(max_lag-aux_2[i,j])
            prb=int(len(u)-aux_2[i,j])
            vec = u[pra:prb]
        return vec
    #=============================================================================================================================================================

    """
    This function returns the regressor matrix from a given dataset and the referencied model object.
        Args:
            reg_code = Matrix with codification of regressor elements
            y = Vec that contains the outputs
            u = Vec that contains the inputs
        Raises:
            w = Regressor matrix from the referencied model object
    """
    def get_regressmatrx(self,reg_code,y,u):
        import numpy as np
        # reg_code=self.reg_code
        [aux_a, aux_b] = reg_code.shape
        w=get_regresvec(self,reg_code,0,0,y,u)
        w_aux=get_regresvec(self,reg_code,0,0,y,u)
        for i in range(1, aux_a):   #The first roll is read outside the loop, becouse that the range starts in 1.
            if aux_b == 1:
                w_aux=get_regresvec(self,reg_code,i,0,y,u)
                w=np.vstack((w,w_aux))
            else:
                for j in range (0,aux_b):
                    if reg_code[i,j] != 0:
                        if j == 0:
                            w_aux=(get_regresvec(self,reg_code,i,j,y,u))*(get_regresvec(self,reg_code,i,j+1,y,u))
                        else:
                            if j == aux_b-1:
                                if reg_code[i,j-1] == 0:
                                    w_aux=get_regresvec(self,reg_code,i,j,y,u)
                            else:
                                if reg_code[i,j-1] == 0:
                                    w_aux=(get_regresvec(self,reg_code,i,j,y,u))*(get_regresvec(self,reg_code,i,j+1,y,u))
                                else:
                                    w_aux=w_aux*(get_regresvec(self,reg_code,i,j+1,y,u))
                w=np.vstack((w,w_aux))
                w_aux=get_regresvec(self,reg_code,0,0,y,u)
        w=w.T
        return  w
    #=============================================================================================================================================================
    """
    This function performes a house transformation

    """

    def house(self,x):
        import numpy as np
        n= len(x)
        u=np.linalg.norm(x,2)
        v = np.array(x)
        aux_a = np.array([1])
        if u != 0:
            aux_b = x[0] + np.sign(x[0])*u
            v = np.array(v[1:n]/aux_b)
            v = np.concatenate((aux_a, v))
        return v
    #=============================================================================================================================================================
    """
    This function performes a rowhouse transformation

    """
    def rowhouse(self,RA, v):
        import numpy as np
        b = -2/(v.T@v)
        # print(b)
        # print('A é',A)
        # print('V é',v)
        w = b*RA.T@v
        m = w.size
        w = w.reshape(1, m)
        m = v.size
        v = v.reshape(m, 1)
        RA = RA+v*w
        B = RA
        return B
    #=============================================================================================================================================================
    """
    This function returns the Error Reduction Ration indice from a given regressor matrix. The results are ordenated from the bigger to the smaller ERR índices.
    It returns also a vec that contains the index of the coluns of the regressor matrix that corresponds a such order. The Psi matrix are given
        Args:
            y = Vec that contains the outputs
            w = Regressor matrix to performe the ERR
            ntp = Process Term Numbers
        Raises:
            err = A vec that contains the ERR indices in order
            piv = A vec that contains the index to put the regressors in order
            psi_final = The Psi matrix
    """
    def ERR(self,y,w,ntp):
        import numpy as np
        sq_y = y[self.max_lag: -1].T@y[self.max_lag: -1]
        A=np.array(w)
        y = np.array([y[self.max_lag:]]).T
        A=np.concatenate((A,y), axis=1)
        [a,b]=A.shape
        piv=np.arange(b-1)
        c = np.zeros(b-1)
        err = np.zeros(b-1)
        for i in np.arange(0,b-1):
            for j in np.arange(i,b-1):
                num = np.array(A[i:a, j].T@A[i:a, b-1])
                num = np.power(num,2)
                den = np.array( (A[i:a, j].T@A[i:a, j]) * sq_y)
                d = num/den
                c[j]=num/den
            c = list(c)
            f = c.index(max(c[i:]))
            err[i] = c[f]
            temp = np.copy(A[:, i])
            A[:, i] = np.copy(A[:, f])
            A[:, f] = np.copy(temp)
            temp2 = np.copy(piv[i])
            piv[i] = np.copy(f)
            piv[f] = np.copy(temp2)
            x = A[i: a, i]
            v = self.house(x)
            aux_1 = A[i: a, i: b]
            row_result = self.rowhouse(aux_1, v)
            A[i: a, i: b] = np.copy(row_result)
            if i ==ntp:
                break
        Piv = piv[0: ntp]
        psi_doidao = np.array(w)
        psi_final = np.copy(psi_doidao[:, Piv])
        reg_code_buffer=self.reg_code
        model_code=np.copy(reg_code_buffer[Piv, :])

        return model_code, err, piv, psi_final
    #=============================================================================================================================================================

    """
    This function estimate the parameters from a model given a psi matrix and a output vec. It uses the Last Square Method and returns a vec that contains
    the estimated parameters.
        Args:
            psi = psi matrix from the model
            y = Vec that contains the outputs
        Raises:
            theta = a vector that contains the estimated parameters
    """
    def last_squares(self,psi,y):
        import numpy as np
        theta= (np.linalg.pinv(psi.T@psi))@psi.T@y[self.max_lag:]
        return theta
    #=============================================================================================================================================================

    """
    This function split the data in identificatioin and validation subsets.
        Args:
            y_path = path from txt file that contains the outputs
            u_path = path from txt file that contains the inputs
            percent = percentage of the data set that is destinated as identification set
        Raises:
            y_ident = output identiification data set
            u_ident = input identiification data set
            y_valid = output validation data set
            u_valid = input validation data set
    """

    def prepare_data(y_path,u_path,porcent):
        import numpy as np
        y = np.loadtxt(y_path)
        u = np.loadtxt(u_path)
        y_size=y.size
        n_ident=round((y_size*porcent)/100)
        y_ident=y[0:n_ident]
        u_ident=u[0:n_ident]
        y_valid=y[n_ident+1:y_size+1]
        u_valid=y[n_ident+1:y_size+1]
        return y_ident, u_ident, y_valid, u_valid
    #============================================================================================================================================================
    
    """
    This function function returns the values of RMSE and MSE from a given simulated output and a real output.
        Args:
            y = Real sistem output
            ysim = Simulated output from a model
        Raises:
            rmse = The RMSE index
            mse = The MSE index
    """
    def validation_index(y, ysim):
        import numpy as np
        num = np.power((np.linalg.norm(y - ysim,2), 2))
        den = y.size
        mse = np.divide(num, den)
        media = np.mean(y)
        num2 = np.sqrt(np.power(np.sum((y-ysim), 2)))
        den2 = np.sqrt(np.power(np.sum((y-media), 2)))
        rmse = np.divide(num2, den2)
        return rmse, mse
    #=========================================================================================================================================================
    """
    This function calculates the autocorrelation function for a given vector
        Args:
            x = Vector to calculate the autocorrelation
    """
    def autocorr(x):
        import numpy as np
        result = np.correlate(x, x, mode='full')
        half_of_simmetry = int(np.floor(result.size/2))
        return result[half_of_simmetry:]

