# -*- coding: utf-8 -*-
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
    global get_regresvec
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
        from itertools import combinations_with_replacement
        import numpy as np
        # regressor's code
        ylagmax = 1001 + ylag
        ulagmax = 2001 + ulag
        y_vec = np.arange(1001, ylagmax)
        u_vec = np.arange(2001, ulagmax)
        reg_aux = np.array([0])
        reg_aux = np.concatenate([reg_aux, y_vec, u_vec])
        # generate the matrix with all possible regressors and retur the number of cols(a) and rols(b)
        reg_code = list(combinations_with_replacement(reg_aux, non_degree))
        reg_code = np.array(reg_code)
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
            index_a=int(max_lag-aux_1[i,j])
            index_b=int(len(y)-aux_1[i,j])
            vec = y[index_a:index_b]
        if aux_2[i,j] > 0 and aux_2[i,j] < 100:
            index_a=int(max_lag-aux_2[i,j])
            index_b=int(len(u)-aux_2[i,j])
            vec = u[index_a:index_b]
        return vec
    #=============================================================================================================================================================

    """
    This function returns the regressor matrix from a given dataset and the referencied model object.
        Args:
            y = Vec that contains the outputs
            u = Vec that contains the inputs
        Raises:
            regress_matrix = Regressor matrix from the referencied model object
    """
    def get_regressmatrx(self,y,u):
        import numpy as np
        reg_code=self.reg_code
        [row_number, col_number] = reg_code.shape
        regress_matrix=get_regresvec(self,reg_code,0,0,y,u)
        regress_matrix_aux=get_regresvec(self,reg_code,0,0,y,u)
        for i in range(1, row_number):   #The first roll is read outside the loop, becouse that the range starts in 1.
            if col_number == 1:
                regress_matrix_aux=get_regresvec(self,reg_code,i,0,y,u)
                regress_matrix=np.vstack((regress_matrix,regress_matrix_aux))
            else:
                for j in range (0,col_number):
                    if reg_code[i,j] != 0:
                        if j == 0:
                            regress_matrix_aux=(get_regresvec(self,reg_code,i,j,y,u))*(get_regresvec(self,reg_code,i,j+1,y,u))
                        else:
                            if j == col_number-1:
                                if reg_code[i,j-1] == 0:
                                    regress_matrix_aux=get_regresvec(self,reg_code,i,j,y,u)
                            else:
                                if reg_code[i,j-1] == 0:
                                    regress_matrix_aux=(get_regresvec(self,reg_code,i,j,y,u))*(get_regresvec(self,reg_code,i,j+1,y,u))
                                else:
                                    regress_matrix_aux=regress_matrix_aux*(get_regresvec(self,reg_code,i,j+1,y,u))
                regress_matrix=np.vstack((regress_matrix,regress_matrix_aux))
                regress_matrix_aux=get_regresvec(self,reg_code,0,0,y,u)
        regress_matrix=regress_matrix.T
        return  regress_matrix
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
    It returns also a vec that contains the index of the coluns of the regressor matrix that corresponds a such order. The Psi matrix are also given
        Args:
            y = Vec that contains the outputs
            regress_matrix = Regressor matrix to performe the ERR
            process_term_number = Process Term Numbers
        Raises:
            err = A vec that contains the ERR indices in order
            piv = A vec that contains the index to put the regressors in order
            psi_final = The Psi matrix
    """
    def ERR(self,y,regress_matrix,process_term_number):
        import numpy as np
        squared_y = y[self.max_lag: -1].T@y[self.max_lag: -1]
        aux_regress_matrix=np.array(regress_matrix)
        y = np.array([y[self.max_lag:]]).T
        aux_regress_matrix=np.concatenate((aux_regress_matrix,y), axis=1)
        [row_number,col_number]=aux_regress_matrix.shape
        piv=np.arange(col_number-1)
        print('inicial')
        print(piv)
        err_aux = np.zeros(col_number-1)
        err = np.zeros(col_number-1)
        print('iterações')
        for i in np.arange(0,col_number-1):
            for j in np.arange(i,col_number-1):
                num = np.array(aux_regress_matrix[i:row_number, j].T@aux_regress_matrix[i:row_number, col_number-1])
                num = np.power(num,2)
                den = np.array( (aux_regress_matrix[i:row_number, j].T@aux_regress_matrix[i:row_number, j]) * squared_y)
                d = num/den
                err_aux[j]=num/den
            err_aux = list(err_aux)
            index = err_aux.index(max(err_aux[i:]))
            err[i] = err_aux[index]
            temp = np.copy(aux_regress_matrix[:, i])
            aux_regress_matrix[:, i] = np.copy(aux_regress_matrix[:, index])
            aux_regress_matrix[:, index] = np.copy(temp)
            temp2 = np.copy(piv[i])
            piv[i] = np.copy(index)
            piv[index] = np.copy(temp2)
            print('index', index , '\tvetor',piv , sep=':\t')
            x = aux_regress_matrix[i: row_number, i]
            v = self.house(x)
            aux_1 = aux_regress_matrix[i: row_number, i: col_number]
            row_result = self.rowhouse(aux_1, v)
            aux_regress_matrix[i: row_number, i: col_number] = np.copy(row_result)
            if i == process_term_number:
                break
        Piv = piv[0: process_term_number]
        print('final')
        print(piv)
        psi_aux = np.array(regress_matrix)
        psi_final = np.copy(psi_aux[:, Piv])
        reg_code_buffer=self.reg_code
        model_code=np.copy(reg_code_buffer[Piv, :])

        return model_code, err, piv, psi_final
    #=========================================================================================================================================================
    
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

    def prepare_data(y_path,u_path,percent):
        import numpy as np
        y = np.loadtxt(y_path)
        u = np.loadtxt(u_path)
        y_size=y.size
        size_ident=round((y_size*percent)/100)
        y_ident=y[0:size_ident]
        u_ident=u[0:size_ident]
        y_valid=y[size_ident+1:y_size+1]
        u_valid=y[size_ident+1:y_size+1]
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
    def validation_index(self,y, ysim):
        import numpy as np
        num = np.power(np.linalg.norm((y - ysim)), 2)
        den = y.size
        mse = np.divide(num, den)
        media = np.mean(y)
        num2 = np.sqrt(np.power(np.sum((y - ysim)), 2))
        den2 = np.sqrt(np.power(np.sum((y-media)), 2))
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
    #=========================================================================================================================================================
    """
    This function returns the values of predicted model
        Args:
            model_elements_count = Matrix with regressor codes
            model_pivot = Vector with regressor order (from ERR)
            y_initial = "max_lag" number of values of output mensured to start recursive process
            entrace_u = Vector with entrace values to be used to simulate the model
            estimated_paramters = Paramters estimated to simulated model
        Rises:
            predicted_values = the values of predicted model
    """
    def model_prediction(self,model_elements_count,model_pivot,y_initial,entrace_u,estimated_paramters):
        import numpy as np
        # import pandas as pd
        if len(y_initial)<self.max_lag:
            raise Exception('Insufficient initial conditions elements!')
        predicted_values = np.zeros((len(entrace_u)))
        predicted_values[0:len(y_initial)] = y_initial
        analised_elements_number = self.max_lag + 1
        effective_pivot_vector = model_pivot[0:len(model_elements_count)]
        for i in range(0,len(entrace_u)-self.max_lag):
            # print(i)
            temporary_regressor_matrix = self.get_regressmatrx(predicted_values[i:i+analised_elements_number],entrace_u[i:i+analised_elements_number])
            # print('Unsorted:')
            # print(pd.DataFrame(temporary_regressor_matrix))
            temporary_regressor_matrix = np.copy(temporary_regressor_matrix[:, effective_pivot_vector])
            # print('Sorted:')
            # print(pd.DataFrame(temporary_regressor_matrix))
            predicted_values[i+self.max_lag] = temporary_regressor_matrix @ estimated_paramters
            # input('Debug: Press ENTER to continue...')
        # print(i)
        return predicted_values
    #=========================================================================================================================================================
    """
    This function returns the values information criterion to determine the size of model
        Args:
            output_y = Measured system output
            input_u = Measured system input
        Rises:
            akaike_vector = Vector with values of akaike's information criterion for models with N terms (where N is the vector position + 1)
    """
    def akaike_information_criterion(self,output_y,input_u):
        import numpy as np
        akaike_vector = np.zeros(len(self.reg_code))
        akaike_vector[:] = float('NaN')
        regressor_matrix = self.get_regressmatrx(output_y,input_u)
        [null, null, model_pivot, regressor_matrix] = self.ERR(output_y, regressor_matrix, len(self.reg_code))
        # regressor_matrix = np.copy(regressor_matrix[:, model_pivot])
        for i in range(0,len(self.reg_code)):
            temporary_estimated_paramters = self.last_squares(regressor_matrix[:,0:i+1],output_y)
            temporary_simulated_output = regressor_matrix[:,0:i+1] @ temporary_estimated_paramters
            temporary_residual = output_y[self.max_lag:] - temporary_simulated_output
            residual_variance = np.var(temporary_residual)
            akaike_vector[i] = len(output_y) * residual_variance + 2*(i+1)
        return akaike_vector
