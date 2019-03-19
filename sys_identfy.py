"""

# Authors:
#           Wilson Rocha Lacerda Junior <wilsonrljr@outlook.com>
#           Luan Pascoal da Costa Andrade <luan_pascoal13@hotmail.com>
#           Samuel Carlos Pessoa Oliveira <samuelcpoliveira@gmail.com>
#           Samir Angelo Milani Martins <martins@ufsj.edu.br>
# License: BSD 3 clause

"""


class sys_identfy:
    import numpy as np
    import pandas as pd
    global get_regresvec
    global genreg
    global get_regresvec
    global max_lag


    def __init__(self,non_degree_,ylag_,ulag_):
        """
        This funcrion create a new object of the type sys_identfy. It is called automatically always that a new
        object is instantiated. It is responsible for create the variables from the object and call the function
        genreg that handle the hard job.
        Args:
            non_degree = Nonlinearity degree
            ylag = Max lag of input
            ulag = Max lag of output

        """

        self.non_degree=non_degree_ # non_degree stores the nonlinearity degree
        self.ylag=ylag_             # ylag stores the maximum output lag
        self.ulag=ulag_             # ulag ylag stores the maximum input lag
        [self.reg_code,self.max_lag]=genreg(non_degree_,ylag_,ulag_) # reg_code stores all possible combinations
    #=============================================================================================================================================================

    def genreg(non_degree, ylag, ulag):
        """ This function generates a codification from all possibles regressors given the maximum lag of the
        input and output.

        Parameters:
        -----------
        non_degree = int
                     the desired maximum nonlinearity degree

        ylag = int
                the maximum lag of output regressors

        ulag =int
                the maximum lag of input regressors

        Returns:
        --------
        max_lag = int
                  This value can be used by another functions

        reg_code = ndarray of int
                    Matrix codification of all possible regressors

        Example of codification 1:
        --------
        The codification is defined as:
        100n = y(k-n)
        200n = u(k-n)
        [100n 100n] = y(k-n)y(k-n)
        [200n 200n] = u(k-n)u(k-n)

        """

        from itertools import combinations_with_replacement
        import numpy as np
        ylagmax = 1001 + ylag
        ulagmax = 2001 + ulag
        y_vec = np.arange(1001, ylagmax)
        u_vec = np.arange(2001, ulagmax)
        reg_aux = np.array([0])
        reg_aux = np.concatenate([reg_aux, y_vec, u_vec])
        reg_code = list(combinations_with_replacement(reg_aux, non_degree))
        reg_code = np.array(reg_code)
        max_lag = max(ylag, ulag)
        return reg_code, max_lag
    #=============================================================================================================================================================

    def get_regresvec(self, reg_code_, i, j, y, u):
        """ This function extrac a givn regressor from a dataset

        Parameters:
        -----------

        reg_code = ndarray of int
                    A ndarray that contains the mappin of all possible regressors (it must be generate with genreg() func. )

        i = int
            Row from the map

        j = int
            Column from the map

        y = array-like
            target data used on training phase

        u = array-like
            input data used on training phase

        Returns:
        --------
        vec = array-like
              A vector that contain the refer regressor
        """

        import numpy as np
        max_lag = self.max_lag
        aux_1 = reg_code_-1000
        aux_2 = reg_code_-2000
        if reg_code_[i, j] == 0:
            vec = np.ones(len(y) - (self.max_lag))
        if aux_1[i, j] > 0 and aux_1[i, j] < 100:
            index_a = int(max_lag - aux_1[i, j])
            index_b = int(len(y) - aux_1[i, j])
            vec = y[index_a:index_b]
        if aux_2[i, j] > 0 and aux_2[i, j] < 100:
            index_a = int(max_lag - aux_2[i, j])
            index_b = int(len(u) - aux_2[i, j])
            vec = u[index_a:index_b]
        return vec
    #=============================================================================================================================================================

    def get_regressmatrx(self, y, u):
        """ Build the information matrix based on model code.

        Parameters:
        -----------
        y = array-like
            target data used on training phase

        u = array-like
            input data used on training phase

        Returns:
        --------
        matrix_of_regressors = ndarray of floats
                                the information matrix of the model

        """

        import numpy as np
        reg_code = self.reg_code
        [row_number, col_number] = reg_code.shape
        matrix_of_regressors = get_regresvec(self, reg_code, 0, 0, y, u)
        regress_matrix_aux = get_regresvec(self, reg_code, 0, 0, y, u)
        for i in range(1, row_number):   #The first roll is read outside the loop, becouse that the range starts in 1.
            if col_number == 1:
                regress_matrix_aux = get_regresvec(self, reg_code, i, 0, y, u)
                matrix_of_regressors = np.vstack((regress_matrix, regress_matrix_aux))
            else:
                for j in range (0, col_number):
                    if reg_code[i, j] != 0:
                        if j == 0:
                            regress_matrix_aux = (get_regresvec(self, reg_code, i, j, y, u))*(get_regresvec(self, reg_code, i, j+1, y, u))
                        else:
                            if j == col_number - 1:
                                if reg_code[i, j-1] == 0:
                                    regress_matrix_aux = get_regresvec(self, reg_code, i, j, y, u)
                            else:
                                if reg_code[i, j-1] == 0:
                                    regress_matrix_aux = (get_regresvec(self, reg_code, i, j, y, u))*(get_regresvec(self, reg_code, i, j+1, y, u))
                                else:
                                    regress_matrix_aux = regress_matrix_aux*(get_regresvec(self, reg_code, i, j+1, y, u))
                matrix_of_regressors = np.vstack((matrix_of_regressors, regress_matrix_aux))
                regress_matrix_aux = get_regresvec(self, reg_code, 0, 0, y, u)
        matrix_of_regressors = matrix_of_regressors.T
        return  matrix_of_regressors
    #=============================================================================================================================================================

    def house(self, x):
        """This function performes a Househoulder reflection of vector

        Parameters:
        -----------
        x = array-like of shape = number_of_training_samples
            The respective column of the matrix of regressors in each iteration of ERR function

        Returns:
        --------
        v = array-like of shape = number_of_training_samples
            the reflection of the array x

        References:
        -----------
        [1]`Wikipedia entry on Householder transformation
            <https://en.wikipedia.org/wiki/Householder_transformation>`_
        """

        import numpy as np
        n = len(x)
        u = np.linalg.norm(x, 2)
        v = np.array(x)
        aux_a = np.array([1])
        if u != 0:
            aux_b = x[0] + np.sign(x[0])*u
            v = np.array(v[1: n]/aux_b)
            v = np.concatenate((aux_a, v))
        return v
    #=============================================================================================================================================================

    def rowhouse(self, RA, v):
        """This function performes a row Househoulder transformation

        Parameters:
        -----------
        RA = array-like of shape = number_of_training_samples
            The respective column of the matrix of regressors in each iteration of ERR function

        v = array-like of shape = number_of_training_samples
            the reflected vector obtained by using the householder reflection

        Returns:
        --------
        B = array-like of shape = number_of_training_samples

        """

        import numpy as np
        b = -2/(v.T@v)
        w = b*RA.T@v
        m = w.size
        w = w.reshape(1, m)
        m = v.size
        v = v.reshape(m, 1)
        RA = RA + v*w
        B = RA
        return B
    #=============================================================================================================================================================

    def ERR(self, y, matrix_of_regressors, process_term_number):

        """ Performs the Error Reduction Ration algorithm

        Parameters:
        -----------
        y = array-like of shape = number_of_samples
            the target data used in the identification process

        matrix_of_regressors = ndarray of floats
                                the information matrix of the model

        process_term_number = int
                              Number of Process Terms defined by the user

        Returns:
        --------
        err = array-like of shape = number_of_model_elements
              The respective ERR calculated for each regressor

        piv = array-like of shape = number_of_model_elements
              Contains the index to put the regressors in the correct order based on err values

        matrix_of_regressors_orthogonal = ndarray of floats
                                        The updated and orthogonal information matrix

        References:
        -----------
        [1]`Manuscript: Orthogonal least squares methods and their application to non-linear system identification
            <https://eprints.soton.ac.uk/251147/1/778742007_content.pdf>`_

        [2]`Manuscript (portuguese): Identificação de Sistemas não Lineares Utilizando Modelos NARMAX Polinomiais–Uma Revisão e Novos Resultados
            <https://www.researchgate.net/profile/Giovani_Rodrigues/publication/228595821_Identificacao_de_Sistemas_nao_Lineares_Utilizando_Modelos_NARMAX_Polinomiais-Uma_Revisao_e_Novos_Resultados/links/00b4951b10ff8ab4d3000000.pdf>`_

        """

        import numpy as np
        squared_y = y[self.max_lag: ].T@y[self.max_lag: ]
        aux_regress_matrix = np.array(matrix_of_regressors)
        y = np.array([y[self.max_lag: ]]).T
        y_aux = np.copy(y)
        [row_number, col_number] = aux_regress_matrix.shape
        piv = np.arange(col_number)
        err_aux = np.zeros(col_number)
        err = np.zeros(col_number)
        for i in np.arange(0, col_number):
            for j in np.arange(i, col_number):
                numerator = np.array(aux_regress_matrix[i: row_number, j].T@y_aux[i: row_number])
                numerator = np.power(numerator, 2)
                denominator = np.array( (aux_regress_matrix[i:row_number, j].T@aux_regress_matrix[i: row_number, j]) * squared_y)
                err_aux[j] = numerator/denominator

            if i == process_term_number:
                break
            err_aux = list(err_aux)
            piv_index = err_aux.index(max(err_aux[i: ]))
            err[i] = err_aux[piv_index]
            aux_regress_matrix[:, [piv_index, i]] = aux_regress_matrix[:, [i, piv_index]]
            piv[[piv_index, i]] = piv[[i, piv_index]]
            x = aux_regress_matrix[i: row_number, i]
            v = self.house(x)
            aux_1 = aux_regress_matrix[i: row_number, i: col_number]
            row_result = self.rowhouse(aux_1, v)
            y_aux[i: row_number] = self.rowhouse(y_aux[i: row_number], v)
            aux_regress_matrix[i: row_number, i: col_number] = np.copy(row_result)

        Piv = piv[0: process_term_number]
        psi_aux = np.array(matrix_of_regressors)
        matrix_of_regressors_orthogonal = np.copy(psi_aux[:, Piv])
        reg_code_buffer = self.reg_code
        model_code = np.copy(reg_code_buffer[Piv, :])

        return model_code, err, piv, matrix_of_regressors_orthogonal

    def last_squares(self, matrix_of_regressors, training_output):
        """ Estimate the model parameters using Least Squares method

        Parameters:
        -----------
        matrix_of_regressors = ndarray of floats
                                the information matrix of the model

        training_output = array-like of shape = y_training
                          the data used to training the model

        Returns:
        --------
        theta = array-like of shape = number_of_model_elements
                The estimated parameters of the model
        """

        import numpy as np
        theta = (np.linalg.pinv(matrix_of_regressors.T@matrix_of_regressors))@matrix_of_regressors.T@training_output[self.max_lag: ]
        return theta


    def prepare_data(y_path, u_path, training_percent):
        """ This function split the data in identificatioin and validation subsets.

        Parameters:
        -----------
        y_path = str
                 path from txt file that contains the outputs

        u_path = str
                 path from txt file that contains the inputs

        training_percent = float
                  percentage of the data set that is destinated as identification set

        Returns:
        --------
        y_training = array-like
                  target data used on training phase

        u_training = array-like
                  input data used on training phase

        y_validation = array-like
                  target data for model validation

        u_validation = array-like
                  input data for model validation
        """

        import numpy as np
        y = np.loadtxt(y_path)
        u = np.loadtxt(u_path)
        y_size = y.size
        size_ident = round((y_size*training_percent)/100)
        y_training = y[0: size_ident]
        u_training = u[0: size_ident]
        y_validation = y[size_ident+1 : y_size+1]
        u_validation = y[size_ident+1 : y_size+1]
        return y_training, u_training, y_validation, u_validation

    def forecast_error(self, y, ysim):
        """ Calculate the forecast error (also known as identification residues)
            in a regression model

        Parameters
        ----------
        y : array-like of shape = number_of_outputs
            Represent the target values.

        ysim : array-like of shape = number_of_outputs
            Target values predicted by the model.

        Returns
        -------
        loss : ndarray of floats
               The difference between the true target values and the predicted or forecast
               value in regression or any other phenomenon.

        References
        ----------
        [1] `Wikipedia entry on the Forecast error
            <https://en.wikipedia.org/wiki/Forecast_error>`_

        Examples
        --------
        >>> y = [3, -0.5, 2, 7]
        >>> ysim = [2.5, 0.0, 2, 8]
        >>> forecast_error(y, ysim)
        [0.5, -0.5, 0, -1]

        """

        import numpy as np
        return y - ysim

    def mean_forecast_error(self, y, ysim):
        """ Calculate the mean of forecast error of a regression model

        Parameters
        ----------
        y : array-like of shape = number_of_outputs
            Represent the target values.

        ysim : array-like of shape = number_of_outputs
            Target values predicted by the model.

        Returns
        -------
        loss : float
               The mean  value of the difference between the true target values and the predicted or forecast
               value in regression or any other phenomenon.

        References
        ----------
        [1] `Wikipedia entry on the Forecast error
            <https://en.wikipedia.org/wiki/Forecast_error>`_

        Examples
        --------
        >>> y = [3, -0.5, 2, 7]
        >>> ysim = [2.5, 0.0, 2, 8]
        >>> mean_forecast_error(y, ysim)
        -0.25

        """
        import numpy as np
        return np.average(y - ysim)

    def mean_squared_error(self, y, ysim):
        """ Calculate the Mean Squared Error in a regression model


        Parameters
        ----------
        y : array-like of shape = number_of_outputs
            Represent the target values.

        ysim : array-like of shape = number_of_outputs
            Target values predicted by the model.

        Returns
        -------
        loss : float
            MSE output is non-negative values. Becoming 0.0 means your
            model outputs are exactly matched by true target values.

        References
        ----------
        [1] `Wikipedia entry on the Mean Squared Error
            <https://en.wikipedia.org/wiki/Mean_squared_error>`_

        Examples
        --------
        >>> y = [3, -0.5, 2, 7]
        >>> ysim = [2.5, 0.0, 2, 8]
        >>> mean_squared_error(y, ysim)
        0.375

        """
        import numpy as np
        output_error = np.average((y - ysim) ** 2)
        return np.average(output_error)

    def root_mean_squared_error(self, y, ysim):
        """ Calculate the Root Mean Squared Error in a regression model


        Parameters
        ----------
        y : array-like of shape = number_of_outputs
            Represent the target values.

        ysim : array-like of shape = number_of_outputs
            Target values predicted by the model.

        Returns
        -------
        loss : float
            RMSE output is non-negative values. Becoming 0.0 means your
            model outputs are exactly matched by true target values.

        References
        ----------
        [1] `Wikipedia entry on the Root Mean Squared Error
            <https://en.wikipedia.org/wiki/Root-mean-square_deviation>`_

        Examples
        --------
        >>> y = [3, -0.5, 2, 7]
        >>> ysim = [2.5, 0.0, 2, 8]
        >>> root_mean_squared_error(y, ysim)
        0.612

        """
        import numpy as np
        return np.sqrt(self.mean_squared_error(y, ysim))

    def normalized_root_mean_squared_error(self, y, ysim):
        """ Calculate the normalized Root Mean Squared Error in a regression model


        Parameters
        ----------
        y : array-like of shape = number_of_outputs
            Represent the target values.

        ysim : array-like of shape = number_of_outputs
            Target values predicted by the model.

        Returns
        -------
        loss : float
            nRMSE output is non-negative values. Becoming 0.0 means your
            model outputs are exactly matched by true target values.

        References
        ----------
        [1] `Wikipedia entry on the normalized Root Mean Squared Error
            <https://en.wikipedia.org/wiki/Root-mean-square_deviation>`_

        Examples
        --------
        >>> y = [3, -0.5, 2, 7]
        >>> ysim = [2.5, 0.0, 2, 8]
        >>> normalized_root_mean_squared_error(y, ysim)
        0.081

        """
        import numpy as np
        return self.root_mean_squared_error(y, ysim) / (y.max() - y.min())


    def root_relative_squared_error(self, y, ysim):
        """ Calculate the Root Relative Mean Squared Error in a regression model


        Parameters
        ----------
        y : array-like of shape = number_of_outputs
            Represent the target values.

        ysim : array-like of shape = number_of_outputs
            Target values predicted by the model.

        Returns
        -------
        loss : float
            RRSE output is non-negative values. Becoming 0.0 means your
            model outputs are exactly matched by true target values.

        Examples
        --------
        >>> y = [3, -0.5, 2, 7]
        >>> ysim = [2.5, 0.0, 2, 8]
        >>> root_relative_mean_squared_error(y, ysim)
        0.226

        """
        import numpy as np
        numerator = np.sum(np.square((ysim - y)))
        denominator = np.sum(np.square((ysim - np.mean(y, axis=0))))
        return np.sqrt(np.divide(numerator, denominator))

    def mean_absolute_error(self, y, ysim):
        """ Calculate the Mean absolute error in a regression model


        Parameters
        ----------
        y : array-like of shape = number_of_outputs
            Represent the target values.

        ysim : array-like of shape = number_of_outputs
            Target values predicted by the model.

        Returns
        -------
        loss : float or ndarray of floats
            MAE output is non-negative values. Becoming 0.0 means your
            model outputs are exactly matched by true target values.

        References
        ----------
        [1] `Wikipedia entry on the Mean absolute error
            <https://en.wikipedia.org/wiki/Mean_absolute_error>`_

        Examples
        --------
        >>> y = [3, -0.5, 2, 7]
        >>> ysim = [2.5, 0.0, 2, 8]
        >>> mean_absolute_error(y, ysim)
        0.5

        """

        import numpy as np
        output_errors = np.average(np.abs(y - ysim))
        return np.average(output_errors)

    def mean_squared_log_error(self, y, ysim):
        """ Calculate the Mean Squared Logarithmic Error in a regression model


        Parameters
        ----------
        y : array-like of shape = number_of_outputs
            Represent the target values.

        ysim : array-like of shape = number_of_outputs
            Target values predicted by the model.

        Returns
        -------
        loss : float
            MSLE output is non-negative values. Becoming 0.0 means your
            model outputs are exactly matched by true target values.

        Examples
        --------
        >>> y = [3, 5, 2.5, 7]
        >>> ysim = [2.5, 5, 4, 8]
        >>> mean_squared_log_error(y, ysim)
        0.039

        """
        import numpy as np
        return self.mean_squared_error(np.log1p(y), np.log1p(ysim))

    def median_absolute_error(self, y, ysim):
        """ Calculate the Median Absolute Error in a regression model


        Parameters
        ----------
        y : array-like of shape = number_of_outputs
            Represent the target values.

        ysim : array-like of shape = number_of_outputs
            Target values predicted by the model.

        Returns
        -------
        loss : float
            MdAE output is non-negative values. Becoming 0.0 means your
            model outputs are exactly matched by true target values.

        References
        ----------
        [1] `Wikipedia entry on the Median absolute deviation
            <https://en.wikipedia.org/wiki/Median_absolute_deviation>`_


        Examples
        --------
         >>> y = [3, -0.5, 2, 7]
        >>> ysim = [2.5, 0.0, 2, 8]
        >>> median_absolute_error(y, ysim)
        0.5

        """
        import numpy as np
        return np.median(np.abs(y - ysim))

    def explained_variance_score(self, y, ysim):
        """ Calculate the Explained Variance Score of a regression model


        Parameters
        ----------
        y : array-like of shape = number_of_outputs
            Represent the target values.

        ysim : array-like of shape = number_of_outputs
            Target values predicted by the model.

        Returns
        -------
        loss : float
            EVS output is non-negative values. Becoming 1.0 means your
            model outputs are exactly matched by true target values.
            Lower values means worse results

        References
        ----------
        [1] `Wikipedia entry on the Explained Variance
            <https://en.wikipedia.org/wiki/Explained_variation>`_


        Examples
        --------
         >>> y = [3, -0.5, 2, 7]
        >>> ysim = [2.5, 0.0, 2, 8]
        >>> explained_variance_score(y, ysim)
        0.957

        """

        import numpy as np
        y_diff_avg = np.average(y - ysim)
        numerator = np.average((y - ysim - y_diff_avg) ** 2)
        y_avg = np.average(y)
        denominator = np.average((y- y_avg) ** 2)
        nonzero_numerator = numerator != 0
        nonzero_denominator = denominator != 0
        valid_score = nonzero_numerator & nonzero_denominator
        output_scores = np.ones(y.shape[0])
        output_scores[valid_score] = 1 - (numerator[valid_score] /
                                      denominator[valid_score])
        output_scores[nonzero_numerator & ~nonzero_denominator] = 0.
        return np.average(output_scores)

    def r2_score(self, y, ysim):
        """ Calculate the R2 score of a regression model


        Parameters
        ----------
        y : array-like of shape = number_of_outputs
            Represent the target values.

        ysim : array-like of shape = number_of_outputs
            Target values predicted by the model.

        Returns
        -------
        loss : float
            R2 output can be non-negative values or negative value. Becoming 1.0 means your
            model outputs are exactly matched by true target values.
            Lower values means worse results

        Notes
        -----
        This is not a symmetric function

        References
        ----------
        [1] `Wikipedia entry on the Coefficient of determination
            <https://en.wikipedia.org/wiki/Coefficient_of_determination>`_


        Examples
        --------
         >>> y = [3, -0.5, 2, 7]
        >>> ysim = [2.5, 0.0, 2, 8]
        >>> explained_variance_score(y, ysim)
        0.948

        """
        import numpy as np
        numerator = ((y - ysim) ** 2).sum(axis=0, dtype=np.float64)
        denominator = ((y - np.average(y, axis=0)) ** 2).sum(axis=0, dtype=np.float64)
        nonzero_denominator = denominator != 0
        nonzero_numerator = numerator != 0
        valid_score = nonzero_denominator & nonzero_numerator
        output_scores = np.ones([y.shape[0]])
        output_scores[valid_score] = 1 - (numerator[valid_score] /
                                      denominator[valid_score])
        # arbitrary set to zero to avoid -inf scores, having a constant
        # y_true is not interesting for scoring a regression anyway
        output_scores[nonzero_numerator & ~nonzero_denominator] = 0.

        return np.average(output_scores)

    def symmetric_mean_absolute_percentage_error(self, y, ysim):
        """ Calculate the SMAPE score of a regression model


        Parameters
        ----------
        y : array-like of shape = number_of_outputs
            Represent the target values.

        ysim : array-like of shape = number_of_outputs
            Target values predicted by the model.

        Returns
        -------
        loss : float
            SMAPE output is a non-negative value.
            The results are percentages values.

        Notes
        -----
        One supposed problem with SMAPE is that it is not symmetric since over-forecasts
        and under-forecasts are not treated equally.

        References
        ----------
        [1] `Wikipedia entry on the Symmetric mean absolute percentage error
            <https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error>`_


        Examples
        --------
         >>> y = [3, -0.5, 2, 7]
        >>> ysim = [2.5, 0.0, 2, 8]
        >>> symmetric_mean_absolute_percentage_error(y, ysim)
        57.87

        """

        import numpy as np
        return 100/len(y) * np.sum(2*np.abs(ysim - y) / (np.abs(y) + np.abs(ysim)))
        #return np.mean((np.abs(ysim - y) * 200/ (np.abs(ysim) + np.abs(y))))


    def autocorr(self, signal):
        """ Performs the autocorrelation of a signal to help the
        user to choose the most adequate decimation factor


        Parameters
        ----------
        signal : array-like of shape = number_of_samples

        Returns
        -------
        autocorrelated_values : ndarray of floats

        References
        ----------
        [1] `Wikipedia entry on the Autocorrelation
            <https://en.wikipedia.org/wiki/Autocorrelation>`_


        Examples
        --------
         >>> y = [3, -0.5, 2, 7]
        >>> autocorr(y)
        [62.25 11.5   2.5  21.  ]

        """

        import numpy as np
        result = np.correlate(signal, signal, mode='full')
        half_of_simmetry = int(np.floor(result.size/2))
        return result[half_of_simmetry: ]
    #=========================================================================================================================================================

    def model_prediction(self, model_elements, model_pivot, y_initial, entrace_u, estimated_paramters):

        """ Performs the free run simulation (infinity steps-ahead simulation) of a model


        Parameters
        ----------
        model_elements = ndarray of ints
                         Matrix with regressor codes

        model_pivot = array-like of shape = number_of_model_elements
                      Vector with regressor order (from ERR)

        y_initial = array-like of shape = max_lag
                    number of initial conditions values of output mensured 
                    to start recursive process

        entrace_u = ndarray of floats of shape = number_of_samples
                    Vector with entrace values to be used in model simulation

        estimated_paramters = array-like of shape = number_of_model_elements
                              Paramters estimated via Least Squares method

        Returns
        -------
        predicted_values = ndarray of floats
                           the predicted values of the model


        """

        import numpy as np
        if len(y_initial)<self.max_lag:
            raise Exception('Insufficient initial conditions elements!')
        predicted_values = np.zeros((len(entrace_u)))
        predicted_values[0:self.max_lag] = y_initial[0:self.max_lag] #Discard unnecessary initial values
        analised_elements_number = self.max_lag + 1
        effective_pivot_vector = model_pivot[0: len(model_elements)]
        for i in range(0, len(entrace_u)-self.max_lag):
            # print(i)
            temporary_regressor_matrix = self.get_regressmatrx(predicted_values[i:i+analised_elements_number],entrace_u[i:i+analised_elements_number])
            temporary_regressor_matrix = np.copy(temporary_regressor_matrix[:, effective_pivot_vector])
            predicted_values[i+self.max_lag] = temporary_regressor_matrix @ estimated_paramters
        return predicted_values

    def information_criterion(self, output_y, input_u, calculation_method):
        """ This function performs a information criterion to determine the model size

        Parameters
        ----------
        output_y = array-like of shape = number_of_samples
                    Target values of the system

        input_u = array-like of shape = number_of_samples
                  Input system values measured by the user

        calculation_method = int value to choose the respective information criteria
                             0 - Akaike's Information Criterion with critical value 2 (AIC) (default)
                             1 - Bayes Information Criterion (BIC)
                             2 - Final Prediction Error (FPE)
                             3 - Khundrin’s law ofiterated logarithm criterion (LILC)

        Returns:
        --------
        output_vector = array-like of shape = number_of_elements
                        Vector with values of akaike's information criterion for models
                        with N terms (where N is the vector position + 1)

        References
        ----------

        """

        import numpy as np
        output_vector = np.zeros(len(self.reg_code))
        output_vector[:] = float('NaN')
        base_regressor_matrix = self.get_regressmatrx(output_y, input_u)
        [null, null, null, regressor_matrix] = self.ERR(output_y, base_regressor_matrix, len(self.reg_code))
        effective_output_elements_count = len(output_y) - self.max_lag

        for i in range(0, len(self.reg_code)):
            model_elements = i + 1
            temporary_estimated_paramters = self.last_squares(regressor_matrix[:, 0:model_elements], output_y)
            temporary_simulated_output = regressor_matrix[:, 0:model_elements] @ temporary_estimated_paramters
            temporary_residual = output_y[self.max_lag: ] - temporary_simulated_output
            residual_variance = np.var(temporary_residual)

            if calculation_method == 1: #BIC
                model_factor = model_elements * np.log(effective_output_elements_count)
            elif calculation_method == 2: #FPE
                model_factor = effective_output_elements_count * np.log((effective_output_elements_count + model_elements) / (effective_output_elements_count - model_elements))
            elif calculation_method == 3: #LILC
                model_factor = 2 * model_elements * np.log(np.log(effective_output_elements_count))
            else: #AIC-2
                model_factor =  + 2 * model_elements

            residual_factor = (effective_output_elements_count)*np.log(residual_variance)
            output_vector[i] = residual_factor + model_factor

        return output_vector
