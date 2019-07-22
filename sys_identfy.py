"""

# Authors:
#           Wilson Rocha Lacerda Junior <wilsonrljr@outlook.com>
#           Luan Pascoal da Costa Andrade <luan_pascoal13@hotmail.com>
#           Samuel Carlos Pessoa Oliveira <samuelcpoliveira@gmail.com>
#           Samir Angelo Milani Martins <martins@ufsj.edu.br>
# License: BSD 3 clause

"""
import numpy
from itertools import combinations_with_replacement
from collections import Counter

class sys_identfy:

    def __init__(self, non_degree=2, ylag=2, xlag=2, info_criteria='aic',
                 scoring='root_relative_squared_error', n_terms=None):
        """
        This funcrion create a new object of the type sys_identfy.
        It is called automatically always that a new object is
        instantiated. It is responsible for create the variables
        from the object and call the function regressors_space that
        handle the hard job.

        Args:
            non_degree = Nonlinearity degree
            ylag = Max lag of input
            xlag = Max lag of output

        """

        self.non_degree = non_degree
        self.ylag = ylag
        self.xlag = xlag
        [self.reg_code, self.max_lag] = self.regressors_space(non_degree, ylag, xlag)
        self.info_criteria = info_criteria
        self.scoring = getattr(self, scoring)
        self.n_terms = n_terms

    def score(self, y, y_predicted):
        return self.scoring(y, y_predicted)

    def results(self, theta_precision=4, err_precision=8):
        """ This function returns model regressors, associated parameter and
        respective ERR value on a string matrix.

        Parameters:
        -----------
        theta_precision = int (default: 4)
                          precision of shown parameters values

        err_precision = int (default: 8)
                        precision of shown ERR values

        Returns:
        --------
        output_matrix = string
            Where:
                first column represents each regressor element;
                second column represents associated parameter;
                third column represents the error reduction ratio associated to each regressor.
        """

        output_matrix = []
        for i in range(0, self.n_terms):
            if numpy.max(self.final_model[i]) < 1:
                actual_regressor = str(1)
            else:
                regressor_dic = Counter(self.final_model[i])
                regressor_string = []
                for j in range(0, len(list(regressor_dic.keys()))):
                    regressor_key = list(regressor_dic.keys())[j]
                    if regressor_key < 1:
                        translated_key = ''
                        translated_exponent = ''
                    else:
                        delay_string = str(int(regressor_key - numpy.floor(regressor_key/1000)*1000))
                        if int(regressor_key/1000) < 2:
                            translated_key = 'y(k-' + delay_string + ')'
                        else:
                            translated_key = 'u' + str(int(regressor_key/1000)-1) + '(k-' + delay_string + ')'
                        if regressor_dic[regressor_key] < 2:
                            translated_exponent = ''
                        else:
                            translated_exponent = '^' + str(regressor_dic[regressor_key])
                    regressor_string.append(translated_key + translated_exponent)
                actual_regressor = ''.join(regressor_string)
            actual_parameter = str(numpy.round(self.theta[i,0], theta_precision))
            actual_err = str(numpy.round(self.err[i], err_precision))
            actual_output = [actual_regressor, actual_parameter, actual_err]
            output_matrix.append(actual_output)
        return output_matrix

    def fit(self, X, y):
        reg_Matrix = self.build_information_matrix(self.reg_code, X, y)
        self.info_values = self.information_criterion(X, y)
        if self.n_terms is None:
            model_length = numpy.where(self.info_values == numpy.amin(self.info_values))
            model_length = int(model_length[0])
            self.n_terms = model_length
        else:
            model_length = self.n_terms
        [model, self.err, self.pivv, psi] = self.error_reduction_ration(y, reg_Matrix, model_length)
        number_of_elements, nno, maximum_lag, number_of_output, nu, self.final_model = self.model_information(model)
        self.theta = self.last_squares(psi, y)
        return self.final_model, self.pivv, self.theta, self.info_values, self.err

    def predict(self, X, y):
        return self.model_prediction(self.final_model,
                                     self.pivv, X, y,
                                     self.theta)

    def regressors_space(self, non_degree, ylag, xlag):
        """ This function generates a codification from all possibles
            regressors given the maximum lag of the input and output.

        Parameters:
        -----------
        non_degree = int
                     the desired maximum nonlinearity degree

        ylag = int
                the maximum lag of output regressors

        xlag =int
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

        ylagmax = 1001 + ylag
        xlagmax = 2001 + xlag
        y_vec = numpy.arange(1001, ylagmax)
        u_vec = numpy.arange(2001, xlagmax)
        reg_aux = numpy.array([0])
        reg_aux = numpy.concatenate([reg_aux, y_vec, u_vec])
        reg_code = list(combinations_with_replacement(reg_aux, non_degree))
        reg_code = numpy.array(reg_code)
        reg_code = reg_code[:, reg_code.shape[1]::-1]
        max_lag = max(ylag, xlag)
        return reg_code, max_lag

    def house(self, x):
        """This function performes a Househoulder reflection of vector

        Parameters:
        -----------
        x = array-like of shape = number_of_training_samples
            The respective column of the matrix of regressors in each
            iteration of ERR function

        Returns:
        --------
        v = array-like of shape = number_of_training_samples
            the reflection of the array x

        References:
        -----------
        [1]`Wikipedia entry on Householder transformation
            <https://en.wikipedia.org/wiki/Householder_transformation>`_
        """

        n = len(x)
        u = numpy.linalg.norm(x, 2)
        v = numpy.array(x)
        aux_a = numpy.array([1])
        if u != 0:
            aux_b = x[0] + numpy.sign(x[0])*u
            v = numpy.array(v[1: n]/aux_b)
            v = numpy.concatenate((aux_a, v))
        return v

    def rowhouse(self, RA, v):
        """This function performes a row Househoulder transformation

        Parameters:
        -----------
        RA = array-like of shape = number_of_training_samples
            The respective column of the matrix of regressors in each
            iteration of ERR function

        v = array-like of shape = number_of_training_samples
            the reflected vector obtained by using the householder reflection

        Returns:
        --------
        B = array-like of shape = number_of_training_samples

        """

        b = -2/(v.T@v)
        w = b*RA.T@v
        m = w.size
        w = w.reshape(1, m)
        m = v.size
        v = v.reshape(m, 1)
        RA = RA + v*w
        B = RA
        return B

    def error_reduction_ration(self, y, X, process_term_number):

        """ Performs the Error Reduction Ration algorithm

        Parameters:
        -----------
        y = array-like of shape = number_of_samples
            the target data used in the identification process

        X = ndarray of floats
                                the information matrix of the model

        process_term_number = int
                              Number of Process Terms defined by the user

        Returns:
        --------
        err = array-like of shape = number_of_model_elements
              The respective ERR calculated for each regressor

        piv = array-like of shape = number_of_model_elements
              Contains the index to put the regressors in the correct order
              based on err values

        X_orthogonal = ndarray of floats
                                        The updated and orthogonal
                                        information matrix

        References:
        -----------
        [1]`Manuscript: Orthogonal least squares methods and their application
            to non-linear system identification
            <https://eprints.soton.ac.uk/251147/1/778742007_content.pdf>`_

        [2]`Manuscript (portuguese): Identificação de Sistemas não Lineares
            Utilizando Modelos NARMAX Polinomiais–Uma Revisão
            e Novos Resultados
            <https://www.researchgate.net/profile/Giovani_Rodrigues/publication/228595821_Identificacao_de_Sistemas_nao_Lineares_Utilizando_Modelos_NARMAX_Polinomiais-Uma_Revisao_e_Novos_Resultados/links/00b4951b10ff8ab4d3000000.pdf>`_

        """

        squared_y = y[self.max_lag:].T@y[self.max_lag:]
        X_aux = numpy.array(X)
        y = numpy.array([y[self.max_lag:, 0]]).T
        y_aux = numpy.copy(y)
        [row_number, col_number] = X_aux.shape
        piv = numpy.arange(col_number)
        err_aux = numpy.zeros(col_number)
        err = numpy.zeros(col_number)
        for i in numpy.arange(0, col_number):
            for j in numpy.arange(i, col_number):
                num = numpy.array(X_aux[i: row_number, j].T@y_aux[i: row_number])
                num = numpy.power(num, 2)
                den = numpy.array((X_aux[i:row_number, j].T@X_aux[i: row_number, j]) * squared_y)
                err_aux[j] = num/den

            if i == process_term_number:
                break
            err_aux = list(err_aux)
            piv_index = err_aux.index(max(err_aux[i:]))
            err[i] = err_aux[piv_index]
            X_aux[:, [piv_index, i]] = X_aux[:, [i, piv_index]]
            piv[[piv_index, i]] = piv[[i, piv_index]]
            # index = err_aux.index(max(err_aux[i: ]))
            # err[i] = err_aux[index]
            # aux_regress_matrix[:, [index, i]] = aux_regress_matrix[:, [i, index]]
            # piv[[index, i]] = piv[[i, index]]
            x = X_aux[i: row_number, i]
            v = self.house(x)
            aux_1 = X_aux[i: row_number, i: col_number]
            row_result = self.rowhouse(aux_1, v)
            y_aux[i: row_number] = self.rowhouse(y_aux[i: row_number], v)
            X_aux[i: row_number, i: col_number] = numpy.copy(row_result)

        Piv = piv[0: process_term_number]
        psi_aux = numpy.array(X)
        X_orthogonal = numpy.copy(psi_aux[:, Piv])
        psi_aux = numpy.array(X)
        reg_code_buffer = self.reg_code
        model_code = numpy.copy(reg_code_buffer[Piv, :])

        return model_code, err, piv, X_orthogonal

    def last_squares(self, X_train, y_train):
        """ Estimate the model parameters using Least Squares method

        Parameters:
        -----------
        X_train = ndarray of floats
                                the information matrix of the model

        y_train = array-like of shape = y_training
                          the data used to training the model

        Returns:
        --------
        theta = array-like of shape = number_of_model_elements
                The estimated parameters of the model
        """

        y_train = y_train[self.max_lag:, 0]
        y_train = numpy.reshape(y_train, (len(y_train), 1))
        theta = (numpy.linalg.pinv(X_train.T@X_train))@X_train.T@y_train
        return theta

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

        result = numpy.correlate(signal, signal, mode='full')
        half_of_simmetry = int(numpy.floor(result.size/2))
        return result[half_of_simmetry:]

    def model_prediction(self, model_elements, model_pivot,
                         entrace_u, y_initial, theta):

        """ Performs the free run simulation (infinity steps-ahead simulation)
            of a model

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

        theta = array-like of shape = number_of_model_elements
                              Paramters estimated via Least Squares method

        Returns
        -------
        predicted_values = ndarray of floats
                           the predicted values of the model

        """

        if len(y_initial) < self.max_lag:
            raise Exception('Insufficient initial conditions elements!')
        predicted_values = numpy.zeros((len(entrace_u), 1))
        # Discard unnecessary initial values
        predicted_values[0:self.max_lag] = y_initial[0:self.max_lag]
        analised_elements_number = self.max_lag + 1
        effective_pivot_vector = model_pivot[0: len(model_elements)]
        for i in range(0, len(entrace_u)-self.max_lag):
            X_temp = self.build_information_matrix(self.reg_code,
                                                   entrace_u[i:i+analised_elements_number],
                                                   predicted_values[i:i+analised_elements_number])
            X_temp = numpy.copy(X_temp[:, effective_pivot_vector])
            a = X_temp @ theta
            predicted_values[i+self.max_lag] = a[:, 0]

        return predicted_values

    def information_criterion(self, input_u, output_y):
        """ This function performs a information criterion to determine the model size

        Parameters
        ----------
        output_y = array-like of shape = number_of_samples
                    Target values of the system

        input_u = array-like of shape = number_of_samples
                  Input system values measured by the user

        calculation_method = int value to choose the respective
                            information criteria

                             'Akaike'-  Akaike's Information Criterion with
                                        critical value 2 (AIC) (default)
                             'Bayes' -  Bayes Information Criterion (BIC)
                             'FPE'   -  Final Prediction Error (FPE)
                             'LILC'  -  Khundrin’s law ofiterated logarithm
                                        criterion (LILC)

        Returns:
        --------
        output_vector = array-like of shape = number_of_elements
                        Vector with values of akaike's information criterion
                        for models with N terms (where N is the
                        vector position + 1)

        References
        ----------

        """

        output_vector = numpy.zeros(len(self.reg_code))
        output_vector[:] = float('NaN')
        X_base = self.build_information_matrix(self.reg_code,
                                               input_u,
                                               output_y)
        effective_output_elements_count = len(output_y) - self.max_lag
        calculation_method = self.info_criteria

        for i in range(0, len(self.reg_code)):
            model_elements = i + 1
            [null, null, null, regressor_matrix] = self.error_reduction_ration(output_y, X_base,
                                                            model_elements)
            temporary_theta = self.last_squares(regressor_matrix, output_y)
            temporary_simulated_output = regressor_matrix @ temporary_theta
            temporary_residual = (output_y[self.max_lag:]
                                  - temporary_simulated_output)
            residual_variance = numpy.var(temporary_residual)

            if calculation_method == 'bic':
                model_factor = (model_elements
                                * numpy.log(effective_output_elements_count))
            elif calculation_method == 'fpe':
                model_factor = effective_output_elements_count * numpy.log((effective_output_elements_count + model_elements) / (effective_output_elements_count - model_elements))
            elif calculation_method == 'lilc':
                model_factor = (2 * model_elements
                                * numpy.log(numpy.log(effective_output_elements_count)))
            else:  # AIC
                model_factor = + 2 * model_elements

            residual_factor = ((effective_output_elements_count)
                               * numpy.log(residual_variance))
            output_vector[i] = residual_factor + model_factor

        return output_vector

    def model_information(self, model):
        """ This function return crucial information about the current model evaluated

        Parameters
        ----------
        model = ndarray of ints
                The model code represetation

        Returns:
        --------
        number_of_elements = int
                             The number of terms of the model

        nno = int
              The number of noise terms

        maximum_lag = int
                      The maximum lag of any regressor of the model

        number_of_output = int
                           The number of outputs of the model

        number_of_inputs = int
                           The number of inputs of the model

        model = ndarray of ints
                The model code represetation


        Examples
        --------
        Example 1:

        >>> model = ([1001,1001]; [2001, 0]; [2002 1001])
        >>> [n_e, nno, max_lag, n_out, n_in, model] = model_information(model)
        n_e = 3
        nno = 0
        max_lag = 2
        n_out = 1
        n_in = 1
        model = ([1001,1001]; [2001, 0]; [2002 1001])

        """

        number_of_elements = model.shape[0]
        number_of_noise = 0  # require future update for NARMAX model

        # the aux_model variable is an array with all terms
        # of the model separated wihtout brackets
        # example: if lag = 2, ylag = 2, xlag = 2, the aux_model
        # results in
        # [1001 1002 2001 2002 1001 1001 1002 1001 2001 1001 2002 1001 1002
        #  1002 2001 1002 2002 1002 2001 2001 2002 2001 2002 2002]
        aux_model = model.reshape(model.shape[0]*model.shape[1], 1)
        aux_model = aux_model[~numpy.all(aux_model == 0, axis=1)]
        aux_model = numpy.array(aux_model).ravel()
        list_of_split_vector = []
        lag = []
        for k in numpy.arange(aux_model.shape[0]):
            remove_bracket = numpy.array(numpy.array2string(aux_model[k], precision=1, separator=''))
            # the split_vector transform each term in auxiliary_vector
            # in separeted digits
            # example: [1001] = [1 0 0 1]
            split_vector = numpy.array([int(d) for d in str(remove_bracket)])
            # the variable 's' is the join of the split_vector from index 1
            # onwards
            # example: [1 0 0 1] = [001] = 1 and [1 0 1 2] = [012] = 12
            s = int(''.join(str(i) for i in split_vector[1:]))
            lag.append(s)
            list_of_split_vector.append(split_vector)

        list_of_split_vector = numpy.array(list_of_split_vector)
        if len(model) != 0 and len(list_of_split_vector) != 0:
            maximum_lag = numpy.max(lag)
            min_term_code = numpy.min(list_of_split_vector[:, 0])
            if min_term_code != 1:
                number_of_output = 0
            else:
                number_of_output = 1

            number_of_inputs = (numpy.max(list_of_split_vector[:, 0])
                                - number_of_output)

        else:
            maximum_lag = 0
            number_of_output = 0
            number_of_inputs = 0

        _model = numpy.copy(model)

        return number_of_elements, number_of_noise, maximum_lag, number_of_output, number_of_inputs, _model

    def shift_column(self, col_to_shift, lag):
        """ This function shift the values corresponding a regressor given
            its respective lag

        Parameters
        ----------
        col_to_shift = array-like of shape = number_of_samples
                       The samples of the input or output

        lag = int
              The respective lag of the regressor

        Returns:
        --------
        col_aux = array-like of shape = number_of_samples
                  The shifted array of the input or output

        Examples
        --------
        Example 1:

        >>> y = [1, 2, 3, 4, 5]
        >>> shift_column(y, 1)
        [0, 1, 2, 3, 4]
        """

        number_of_samples = col_to_shift.shape[0]
        col_aux = numpy.zeros((number_of_samples, 1))
        aux = col_to_shift[0: number_of_samples - lag]
        aux = numpy.reshape(aux, (len(aux), 1))
        col_aux[lag:, 0] = aux[:, 0]

        return col_aux

    def build_information_matrix(self, model, u, y):
        """ Build the information matrix based on model code.

        Parameters:
        -----------
        model = ndarray of int
                the model code representation

        y = array-like
            target data used on training phase

        u = array-like
            input data used on training phase

        Returns:
        --------
        X = ndarray of floats
                                the information matrix of the model

        """

        number_of_elements = self.model_information(model)
        number_of_samples = u.shape[0]
        auxiliary_model = numpy.copy(model)
        X = numpy.ones((number_of_samples, number_of_elements[0]))
        for k in numpy.arange(number_of_elements[0]):
            current_regressor = auxiliary_model[k, :]
            current_regressor = current_regressor.reshape(current_regressor.shape[0], 1)
            current_regressor = current_regressor[~numpy.all(current_regressor == 0, axis=1)]
            current_regressor = numpy.array(current_regressor).ravel()

            if len(current_regressor) != 0:
                list_of_split_vector = []
                lag = []
                for i in numpy.arange(current_regressor.shape[0]):
                    remove_bracket = numpy.array(numpy.array2string(current_regressor[i], precision=1, separator=''))
                    split_vector = numpy.array([int(d) for d in str(remove_bracket)])
                    s = int(''.join(str(i) for i in split_vector[1:]))
                    lag.append(s)
                    list_of_split_vector.append(split_vector)

                list_of_split_vector = numpy.array(list_of_split_vector)
                lag = numpy.array(lag)
                which_regressor = list_of_split_vector[:, 0]

                for i in numpy.arange(which_regressor.shape[0]):
                    if which_regressor[i] == 1:
                        aux_col = self.shift_column(y, lag[i]).T
                        element_wise_multiplication = numpy.multiply(aux_col, X[:, k]).T
                        X[:, k] = element_wise_multiplication[:, 0]
                    else:
                        auxw = which_regressor[i]-2
                        aux_col = self.shift_column(u[:, auxw], lag[i]).T
                        element_wise_multiplication = numpy.multiply(aux_col, X[:, k]).T
                        X[:, k] = element_wise_multiplication[:, 0]
            else:
                X[:, k] = numpy.ones((number_of_samples, 1))[:, 0]

        X = X[number_of_elements[2]:, :]

        return X

    def prepare_data(self, y_path, u_path, training_percent):
        """ This function split the data in identificatioin and validation subsets.

        Parameters:
        -----------
        y_path = str
                 path from txt file that contains the outputs

        u_path = str
                 path from txt file that contains the inputs

        training_percent = float
                  percentage of the data set that is destinated as
                  identification set

        Returns:
        --------            #print(a)

        y_training = array-like
                  target data used on training phase

        u_training = array-like
                  input data used on training phase

        y_validation = array-like
                  target data for model validation

        u_validation = array-like
                  input data for model validation
        """

        y = numpy.loadtxt(y_path)
        u = numpy.loadtxt(u_path)
        y_size = y.size
        size_ident = round((y_size*training_percent)/100)
        y_training = y[0: size_ident]
        u_training = u[0: size_ident]
        y_validation = y[size_ident+1:y_size+1]
        u_validation = y[size_ident+1:y_size+1]
        return y_training, u_training, y_validation, u_validation

    def forecast_error(self, y, y_predicted):
        """ Calculate the forecast error (also known as identification residues)
            in a regression model

        Parameters
        ----------
        y : array-like of shape = number_of_outputs
            Represent the target values.

        y_predicted : array-like of shape = number_of_outputs
            Target values predicted by the model.

        Returns
        -------
        loss : ndarray of floats
               The difference between the true target values and the predicted
               or forecast value in regression or any other phenomenon.

        References
        ----------
        [1] `Wikipedia entry on the Forecast error
            <https://en.wikipedia.org/wiki/Forecast_error>`_

        Examples
        --------
        >>> y = [3, -0.5, 2, 7]
        >>> y_predicted = [2.5, 0.0, 2, 8]
        >>> forecast_error(y, y_predicted)
        [0.5, -0.5, 0, -1]

        """

        return y - y_predicted

    def mean_forecast_error(self, y, y_predicted):
        """ Calculate the mean of forecast error of a regression model

        Parameters
        ----------
        y : array-like of shape = number_of_outputs
            Represent the target values.

        y_predicted : array-like of shape = number_of_outputs
            Target values predicted by the model.

        Returns
        -------
        loss : float
               The mean  value of the difference between the true target
               values and the predicted or forecast value in regression
               or any other phenomenon.

        References
        ----------
        [1] `Wikipedia entry on the Forecast error
            <https://en.wikipedia.org/wiki/Forecast_error>`_

        Examples
        --------
        >>> y = [3, -0.5, 2, 7]
        >>> y_predicted = [2.5, 0.0, 2, 8]
        >>> mean_forecast_error(y, y_predicted)
        -0.25

        """
        return numpy.average(y - y_predicted)

    def mean_squared_error(self, y, y_predicted):
        """ Calculate the Mean Squared Error in a regression model

        Parameters
        ----------
        y : array-like of shape = number_of_outputs
            Represent the target values.

        y_predicted : array-like of shape = number_of_outputs
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
        >>> y_predicted = [2.5, 0.0, 2, 8]
        >>> mean_squared_error(y, y_predicted)
        0.375

        """
        output_error = numpy.average((y - y_predicted) ** 2)
        return numpy.average(output_error)

    def root_mean_squared_error(self, y, y_predicted):
        """ Calculate the Root Mean Squared Error in a regression model

        Parameters
        ----------
        y : array-like of shape = number_of_outputs
            Represent the target values.

        y_predicted : array-like of shape = number_of_outputs
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
        >>> y_predicted = [2.5, 0.0, 2, 8]
        >>> root_mean_squared_error(y, y_predicted)
        0.612

        """
        return numpy.sqrt(self.mean_squared_error(y, y_predicted))

    def normalized_root_mean_squared_error(self, y, y_predicted):
        """ Calculate the normalized Root Mean Squared Error in a regression model

        Parameters
        ----------
        y : array-like of shape = number_of_outputs
            Represent the target values.

        y_predicted : array-like of shape = number_of_outputs
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
        >>> y_predicted = [2.5, 0.0, 2, 8]
        >>> normalized_root_mean_squared_error(y, y_predicted)
        0.081

        """
        return self.root_mean_squared_error(y, y_predicted) / (y.max() - y.min())

    def root_relative_squared_error(self, y, y_predicted):
        """ Calculate the Root Relative Mean Squared Error in a regression model

        Parameters
        ----------
        y : array-like of shape = number_of_outputs
            Represent the target values.

        y_predicted : array-like of shape = number_of_outputs
            Target values predicted by the model.

        Returns
        -------
        loss : float
            RRSE output is non-negative values. Becoming 0.0 means your
            model outputs are exactly matched by true target values.

        Examples
        --------
        >>> y = [3, -0.5, 2, 7]
        >>> y_predicted = [2.5, 0.0, 2, 8]
        >>> root_relative_mean_squared_error(y, y_predicted)
        0.226

        """
        numerator = numpy.sum(numpy.square((y_predicted - y)))
        denominator = numpy.sum(numpy.square((y_predicted - numpy.mean(y, axis=0))))
        return numpy.sqrt(numpy.divide(numerator, denominator))

    def mean_absolute_error(self, y, y_predicted):
        """ Calculate the Mean absolute error in a regression model

        Parameters
        ----------
        y : array-like of shape = number_of_outputs
            Represent the target values.

        y_predicted : array-like of shape = number_of_outputs
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
        >>> y_predicted = [2.5, 0.0, 2, 8]
        >>> mean_absolute_error(y, y_predicted)
        0.5

        """

        output_errors = numpy.average(numpy.abs(y - y_predicted))
        return numpy.average(output_errors)

    def mean_squared_log_error(self, y, y_predicted):
        """ Calculate the Mean Squared Logarithmic Error in a regression model

        Parameters
        ----------
        y : array-like of shape = number_of_outputs
            Represent the target values.

        y_predicted : array-like of shape = number_of_outputs
            Target values predicted by the model.

        Returns
        -------
        loss : float
            MSLE output is non-negative values. Becoming 0.0 means your
            model outputs are exactly matched by true target values.

        Examples
        --------
        >>> y = [3, 5, 2.5, 7]
        >>> y_predicted = [2.5, 5, 4, 8]
        >>> mean_squared_log_error(y, y_predicted)
        0.039

        """
        return self.mean_squared_error(numpy.log1p(y), numpy.log1p(y_predicted))

    def median_absolute_error(self, y, y_predicted):
        """ Calculate the Median Absolute Error in a regression model

        Parameters
        ----------
        y : array-like of shape = number_of_outputs
            Represent the target values.

        y_predicted : array-like of shape = number_of_outputs
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
        >>> y_predicted = [2.5, 0.0, 2, 8]
        >>> median_absolute_error(y, y_predicted)
        0.5

        """
        return numpy.median(numpy.abs(y - y_predicted))

    def explained_variance_score(self, y, y_predicted):
        """ Calculate the Explained Variance Score of a regression model

        Parameters
        ----------
        y : array-like of shape = number_of_outputs
            Represent the target values.

        y_predicted : array-like of shape = number_of_outputs
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
        >>> y_predicted = [2.5, 0.0, 2, 8]
        >>> explained_variance_score(y, y_predicted)
        0.957

        """

        y_diff_avg = numpy.average(y - y_predicted)
        numerator = numpy.average((y - y_predicted - y_diff_avg) ** 2)
        y_avg = numpy.average(y)
        denominator = numpy.average((y-y_avg)**2)
        nonzero_numerator = numerator != 0
        nonzero_denominator = denominator != 0
        valid_score = nonzero_numerator & nonzero_denominator
        output_scores = numpy.ones(y.shape[0])
        output_scores[valid_score] = (1 - (numerator[valid_score] /
                                      denominator[valid_score]))
        output_scores[nonzero_numerator & ~nonzero_denominator] = 0.
        return numpy.average(output_scores)

    def r2_score(self, y, y_predicted):
        """ Calculate the R2 score of a regression model

        Parameters
        ----------
        y : array-like of shape = number_of_outputs
            Represent the target values.

        y_predicted : array-like of shape = number_of_outputs
            Target values predicted by the model.

        Returns
        -------
        loss : float
            R2 output can be non-negative values or negative value.
            Becoming 1.0 means your model outputs are exactly
            matched by true target values. Lower values means worse results

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
        >>> y_predicted = [2.5, 0.0, 2, 8]
        >>> explained_variance_score(y, y_predicted)
        0.948

        """
        numerator = ((y - y_predicted) ** 2).sum(axis=0, dtype=numpy.float64)
        denominator = ((y - numpy.average(y, axis=0)) ** 2).sum(axis=0, dtype=numpy.float64)
        nonzero_denominator = denominator != 0
        nonzero_numerator = numerator != 0
        valid_score = nonzero_denominator & nonzero_numerator
        output_scores = numpy.ones([y.shape[0]])
        output_scores[valid_score] = (1 - (numerator[valid_score] /
                                      denominator[valid_score]))
        # arbitrary set to zero to avoid -inf scores, having a constant
        # y_true is not interesting for scoring a regression anyway
        output_scores[nonzero_numerator & ~nonzero_denominator] = 0.

        return numpy.average(output_scores)

    def symmetric_mean_absolute_percentage_error(self, y, y_predicted):
        """ Calculate the SMAPE score of a regression model

        Parameters
        ----------
        y : array-like of shape = number_of_outputs
            Represent the target values.

        y_predicted : array-like of shape = number_of_outputs
            Target values predicted by the model.

        Returns
        -------
        loss : float
            SMAPE output is a non-negative value.
            The results are percentages values.

        Notes
        -----
        One supposed problem with SMAPE is that it is not symmetric since
        over-forecasts and under-forecasts are not treated equally.

        References
        ----------
        [1] `Wikipedia entry on the Symmetric mean absolute percentage error
            <https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error>`_


        Examples
        --------
         >>> y = [3, -0.5, 2, 7]
        >>> y_predicted = [2.5, 0.0, 2, 8]
        >>> symmetric_mean_absolute_percentage_error(y, y_predicted)
        57.87

        """

        return 100/len(y) * numpy.sum(2*numpy.abs(y_predicted - y) / (numpy.abs(y) + numpy.abs(y_predicted)))
        # return numpy.mean((numpy.abs(y_predicted - y) * 200/ (numpy.abs(y_predicted) + numpy.abs(y))))
