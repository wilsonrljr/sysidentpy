# Changes in SysIdentPy

## v0.5.3

### CONTRIBUTORS

- wilsonrljr

### CHANGES

IMPORTANT! This update addresses a bug related to the Bilinear basis function for models with more the 2 inputs. This release keep providing crucial groundwork for the future development of SysIdentPy, making easier to add new features and improve the code, setting the stage for a robust and feature-complete 1.0.0 release in the feature.

- **API Changes:**
  - Fix Bilinear basis function issue for models with more than 2 inputs. This fix the `get_max_xlag` method in `basis_function_base` and also fix how `combination_xlag` is created.

## v0.5.2

### CONTRIBUTORS

- wilsonrljr

### CHANGES

IMPORTANT! This update addresses a critical bug related to the Polynomial and Bilinear basis function for models with more the 3 inputs. The issue raised due to the changes in basis function for v0.5.0, but has now been resolved. This release keep providing crucial groundwork for the future development of SysIdentPy, making easier to add new features and improve the code, setting the stage for a robust and feature-complete 1.0.0 release in the feature.

- **API Changes:**
  - Fix Polynomial and Bilinear basis function issue for models with more than 3 inputs.


## v0.5.1

### CONTRIBUTORS

- wilsonrljr

### CHANGES

This update addresses a critical bug related to the unbiased estimator. The issue previously impacted all basis functions but has now been resolved. This release keep providing crucial groundwork for the future development of SysIdentPy, making easier to add new features and improve the code, setting the stage for a robust and feature-complete 1.0.0 release in the feature.

- **Documentation:**
  - Remove unnecessary code when importing basis functions in many examples.

- **API Changes:**
  - Fix unbiased estimator for every basis function.

## v0.5.0

### CONTRIBUTORS

- wilsonrljr
- nataliakeles
- LeWerner42
- Suyash Gaikwad


### CHANGES

This update introduces major new features and important bug fixes. This release provides crucial groundwork for the future development of SysIdentPy, making easier to add new features and improve the code, setting the stage for a robust and feature-complete 1.0.0 release in the feature.


- **New Features:**
  - **MAJOR**: Add Bilinear Basis Function (thanks nataliakeles). Now the user can use Bilinear NARX models for forecasting.
  - **MAJOR**: Add Legendre polynomial basis function. Now the user can use Legendre NARX models for forecasting.
  - **MAJOR**: Add Hermite polynomial basis function. Now the user can use Hermite NARX models for forecasting.
  **MAJOR**: Add Hermite Normalized polynomial basis function. Now the user can use Hermite Normalized NARX models for forecasting.
  **MAJOR**: Add Laguerre polynomial basis function. Now the user can use Laguerre NARX models for forecasting.

- **Documentation:**
  - Add basis function overview.
  - Files related to v.3.* doc removed.
  - Improved formatting in mathematical equations.
  - Fixed typos and grammatical errors in README.md (thanks Suyash Gaikwad and LeWerner42)
  - Minor additions and grammar fixes.
  - Remove book assets from main repository. The assets were moved to sysidentpy-data repository to keep main repository cleaner and lighter.
  - Fixed link in the book cover to ensure it correctly redirects to the book details. Also change x2_val to x_valid in examples of how to use in readme.
  - Add Pix method as an alternative for brazilian sponsors.
  - Fix code documentation for basis function (it was not showing up in the docs before).
  - Remove `pip install` from the list of the dependencies needed in the chapter.

- **Datasets:**
  - Datasets are now available in a separate repository.

- **API Changes:**
  - add deprecated messages for bias and n in Bersntein basis function. Both parameters will be removed in v0.6.0. Use `include_bias` and `degree`, respectively, instead.
  - Deploy-docs.yml: Change option to make a clean build of the documentation.
  - Deploy-docs.yml: Change python version to deploy docs.
  - Support for Python 3.13 depending on the release of the Pytorch 2.6. Every method in sysidentpy works in python 3.13 excluding neural narx.
  - Update mkdocstrings dependency version
  - Change Polynomial check from class name to isinstance method in every class.
  - Remove support for torch==2.4.0 due to pip error in pytorch side. I'll check if it was solved before allow newer versions of pytorch.
  - Make "main" the new default branch. Master branch removed.
  - Change actions from master to main branch.
  - Split basis function classes into multiples files (one for each basis).
  - Fix redundant bias check on bersntein basis.
  - Fix docstring math notation in basis functions docstring.
  - Remove requirements.txt file.
  - Extensive code refactoring, including type hint improvements, docstring enhancements, removal of unused code, and other behind-the-scenes changes to support new features.
  - Add model_type in basis function base fit and predict method.
  - Change variable name from `combinations` to `combination_list` to avoid any issue with itertools `combination` method in case I want to use it in the future.
  - Remove requirements.txt file.


## v0.4.0

### CONTRIBUTORS

- wilsonrljr

### CHANGES

This update introduces several major features and changes, including some breaking changes. There is a guide to help you update your code to the new version. Depending on your model definition, you might not need to change anything. I decided to go directly to version v0.4.0 instead of providing incremental updates (0.3.5, 0.3.6, etc.) because the breaking changes are easy to fix and the new features are highly beneficial. This release provides crucial groundwork for the future development of SysIdentPy, making easier to add new features and improve the code, setting the stage for a robust and feature-complete 1.0.0 release in the feature.


- **New Features:**
  - **MAJOR**: NonNegative Least Squares algorithm for parameter estimation.
  - **MAJOR**: Bounded Variables Least Squares algorithm for parameter estimation.
  - **MAJOR**: Least Squares Minimal Residual algorithm for parameter estimation.
  - **MAJOR**: Error Reduction Ratio algorithm enhancement for FROLS model structure selection. Users can now set an `err_tol` value to stop the algorithm when the sum of the ERR values reaches this threshold, offering a faster alternative to Information Criteria algorithms. A new example is available in the documentation.
  - **MAJOR**: New Bernstein basis function available, allowing users to choose between Polynomial, Fourier, and Bernstein.
  - **MAJOR**: v0.1 of the companion book "Nonlinear System Identification: Theory and Practice With SysIdentPy." This open-source book serves as robust documentation for the SysIdentPy package and a friendly introduction to Nonlinear System Identification and Timeseries Forecasting. There are case studies in the book that were not included in the documentation at the time of the update release. The book will always feature more in-depth studies and will be updated regularly with additional case studies.

- **Documentation:**
  - All examples updated to reflect changes in v0.4.0.
  - Added guide on defining a custom parameter estimation method and integrating it with SysIdentPy.
  - Documentation moved to the `gh-pages` branch.
  - Defined a GitHub Action to automatically build the docs when changes are pushed to the main branch.
  - Removal of unused code in general

- **Datasets:**
  - Datasets are now available in a separate repository.

- **API Changes:**
  - **BREAKING CHANGE**: Parameter estimation method must now be imported and passed to the model definition, replacing the previous string method. For example, use `from sysidentpy.parameter_estimation import LeastSquares` instead of `"least_squares"`. This change enhances code flexibility, organization, readability, and facilitates easier integration of custom methods. A specific doc page is available to guide migration from v0.3.4 to v0.4.0.
  - **BREAKING CHANGE**: The `fit` method in MetaMSS now requires only `X` and `y` values, omitting the need to pass `fit(X=, y=, X_test=, y_test=)`.
  - Introduced `test_size` hyperparameter to set the proportion of training data used in the fitting process.
  - Added support for Python 3.12.
  - Extensive code refactoring, including type hint improvements, docstring enhancements, removal of unused code, and other behind-the-scenes changes to support new features.

## v0.3.4

### CONTRIBUTORS

- wilsonrljr
- dj-gauthier
- mtsousa

### CHANGES

- **New Features:**
  - **MAJOR**: Ridge Regression Parameter Estimation:
    - Introducing Ridge algorithm for model parameter estimation (Issue #104). Set `estimator="ridge_regression"` and control regularization with the `alpha` parameter. Special thanks to @dj-gauthier and @mtsousa for their contribution. Users are encouraged to visit https://www.researchgate.net/publication/380429918_Controlling_chaos_using_edge_computing_hardware to explore how @dj-gauthier used SysIdentPy in his research.

- **API Changes:**
  - Improved `plotting.py` code with type hints and new options for plotting results.
  - Refactored methods to resolve future warnings from numpy.
  - Code refactoring following PEP8 guidelines.
  - Set "default" as the default style for plotting to avoid errors in new versions of matplotlib.

- **Datasets:**
  - Added `buck_id.csv` and `buck_valid.csv` datasets to the SysIdentPy repository.

- **Documentation:**
  - Add NFIR example (Issue #103). The notebook show how to build models without past output regressors (using only input regressors).
  - Enhanced usage example for MetaMSS.
  - Continued adding type hints to methods.
  - Improved docstrings throughout the codebase.
  - Minor additions and grammar fixes in documentation.
  - @dj-gauthier provided valuable suggestions for enhancing the documentation, which are currently undergoing refinement and will soon be accessible.

- **Development Tools:**
  - Added pre-commit hooks to the repository.
  - Enhanced `pyproject.toml` to assist contributors in setting up their own environment.

## v0.3.3

### CONTRIBUTORS

- wilsonrljr
- GabrielBuenoLeandro
- samirmartins

### CHANGES

- The update **v0.3.3**  has been released with additional features, API changes and fixes.

- MAJOR: Multiobjective Framework: Affine Information Least Squares Algorithm (AILS)
    - Now you can use AILS to estimate parameters of NARMAX models (and variants) using a multiobjective approach.
    - AILS can be accessed using `from sysidentpy.multiobjective_parameter_estimation import AILS`
    - See the docs for a more in depth explanation of how to use AILS.
    - This feature is related to Issue #101. This work is the result of an undergraduate research conducted by Gabriel Bueno Leandro under the supervision of Samir Milani Martins and Wilson Rocha Lacerda Junior.
    - Several new methods were implemented to get the new feature and you can check all of it in sysidentpy -> multiobjective_parameter_estimation.

- API Change: `regressor_code` variable was renamed as `enconding` to avoid using the same name as the method in `narmax_tool` `regressor_code` method.

- DATASET: Added buck_id.csv and buck_valid.csv dataset to SysIdentPy repository.

- DOC: Add a Multiobjetive Parameter Optimization Notebook showing how to use the new AILS method

- DOC: Minor additions and grammar fixes.

## v0.3.2

### CONTRIBUTORS

- wilsonrljr

### CHANGES

- The update **v0.3.2**  has been released with API changes and fixes.

- Major:
    - Added Akaike Information Criteria corrected in FROLS. Now the user can use aicc as the information criteria to select the model order when using FROLS algorithm.

- FIX: Issue #114. Replace yhat with y in root relative squared error. Thanks @miroder

- TESTS: Minor changes in tests by removing unnecessary data load.

- Remove unused code and comments.

- Docs: Minor changes in notebooks. Added AICc method in the information criteria example.

## v0.3.1

### CONTRIBUTORS

- wilsonrljr

### CHANGES

- The update **v0.3.1**  has been released with API changes and fixes.

- API Change:
    - MetaMSS was returning the max lag of the final model instead of the maximum lag related to the xlag and ylag. This is not wrong (its related to the issue #55), but this change will be made for all methods at the same time. In this respect, I'm reverted this to return the maximum lag of the xlag and ylag.

- API Change: Added build_matrix method in BaseMSS. This change improved overall code readability by rewriting if/elif/else clauses in every model structure selection algorithm.

- API Change: Added bic, aic, fpe, and lilc methods in FROLS. Now the method is selected by using a predefined dictionary with the available options. This change improved overall code readability by rewriting if/elif/else clauses in the FROLS algorithm.

- TESTS: Added tests for Neural NARX class. The issue with pytorch was fixed and now we have the tests for every model class.

- Remove unused code and comments.


## v0.3.0

### CONTRIBUTORS

- wilsonrljr
- gamcorn
- Gabo-Tor

### CHANGES

- The update **v0.3.0**  has been released with additional features, API changes and fixes.

- MAJOR: Estimators support in AOLS
    - Now you can use any SysIdentPy estimator in AOLS model structure selection.

- API Change:
    - Refactored base class for model structure selection. A refactored base class for model structure selection has been introduced in SysIdentPy. This update aims to enhance the system identification process by preparing the package for new features that are currently in development, like multiobjective parameter estimation, new basis functions and more.

    Several methods within the base class have undergone significant restructuring to improve their functionality and optimize their performance. This reorganization will facilitate the incorporation of advanced model selection techniques in the future, which will enable users to obtain dynamic models with robust dynamic and static performance.
    - Avoid unnecessary inheritance in every MSS method and improve the readability with better structured classes.
    - Rewritten methods to avoid code duplication.
    - Improve overall code readability by rewriting if/elif/else clauses.

- Breaking Change: `X_train` and `y_train` were replaced respectively by `X` and `y` in `fit` method in MetaMSS model structure selection algorithm.  `X_test` and `y_test` were replaced by `X` and `y` in `predict` method in MetaMSS.

- API Change: Added BaseBasisFunction class, an abstract base class for implementing basis functions.

- Enhancement: Added support for python 3.11.

- Future Deprecation Warning: The user will have to define the estimator and pass it to every model structure selection algorithm instead of using a string to define the Estimator. Currently the estimator is defined like "estimator='least_squares'". In version 0.4.0 the definition will be like "estimator=LeastSquares()"

- FIX: Issue #96. Fix issue with numpy 1.24.* version. Thanks for the contribution @gamcorn.

- FIX: Issue #91. Fix r2_score metric issue with 2 dimensional arrays.

- FIX: Issue #90.

- FIX: Issue #88 .Fix one step ahead prediction error in SimulateNARMAX class (thanks for pointing out, Lalith).

- FIX: Fix error in selecting the correct regressors in AOLS.

- Fix: Fix n step ahead prediction method not returning all values of the defined steps-ahead value when passing only the initial condition.

- FIX: Fix Visible Deprecation Warning raised in get_max_lag method.

- FIX: Fix deprecation warning in Extended Least Squares Example

- DATASET: Added air passengers dataset to SysIdentPy repository.

- DATASET: Added San Francisco Hospital Load dataset to SysIdentPy repository.

- DATASET: Added San Francisco PV GHI dataset to SysIdentPy repository.

- DOC: Improved documentation in Setting Specif Lags page. Now we bring an example of how to set specific lags for MISO models.

- DOC: Minor additions and grammar fixes.

- DOC: Improve image visualization using mkdocs-glightbox.

- Update dev packages versions

## v0.2.1

### CONTRIBUTORS

- wilsonrljr

### CHANGES

- The update **v0.2.1**  has been released with additional feature, minor API changes and fixes.

- MAJOR: Neural NARX now support CUDA
    - Now the user can build Neural NARX models with CUDA support. Just add `device='cuda'` to use the GPU benefits.
    - Updated docs to show how to use the new feature.


- MAJOR: New documentation website
    - The documentation is now entirely based on Markdown (no rst anymore).
    - We use MkDocs and Material for MkDocs theme now.
    - Dark theme option.
    - The Contribute page have more details to help those who wants to contribute with SysIdentPy.
    - New sections (e.g., Blog, Sponsors, etc.)
    - Many improvements under the hood.

- MAJOR: Github Sponsor
    - Now you can support SysIdentPy by becoming a Sponsor! Details: https://github.com/sponsors/wilsonrljr

- Tests:
    - Now there are test for almost every function.
    - Neural NARX tests are raising numpy issues. It'll be fixed til next update.

- FIX: NFIR models in General Estimators
    - Fix support for NFIR models using sklearn estimators.

- The setup is now handled by the pyproject.toml file.

- Remove unused code.

- Fix docstring variables.

- Fix code format issues.

- Fix minor grammatical and spelling mistakes.

- Fix issues related to html on Jupyter notebooks examples on documentation.

- Updated Readme.


## v0.2.0

### CONTRIBUTORS

- wilsonrljr

### CHANGES

- The update **v0.2.0**  has been released with additional feature, minor API changes and fixes.

- MAJOR: Many new features for General Estimators
    - Now the user can build General NARX models with Fourier basis function.
    - The user can choose which basis they want by importing it from sysidentpy.basis_function. Check the notebooks with examples of how to use it.
    - Now it is possible to build General NAR models. The user just need to pass model_type="NAR" to build NAR models.
    - Now it is possible to build General NFIR models. The user just need to pass model_type="NFIR" to build NAR models.
    - Now it is possible to run n-steps ahead prediction using General Estimators. Until now only infinity-steps ahead were allowed. Now the users can set any steps they want.
    - Polynomial and Fourier are supported for now. New basis functions will be added in next releases.
    - No need to pass the number of inputs anymore.
    - Improved docstring.
    - Fixed minor grammatical and spelling mistakes.
    - many under the hood changes.

- MAJOR: Many new features for NARX Neural Network
    - Now the user can build Neural NARX models with Fourier basis function.
    - The user can choose which basis they want by importing it from sysidentpy.basis_function. Check the notebooks with examples of how to use it.
    - Now it is possible to build Neural NAR models. The user just need to pass model_type="NAR" to build NAR models.
    - Now it is possible to build Neural NFIR models. The user just need to pass model_type="NFIR" to build NAR models.
    - Now it is possible to run n-steps ahead prediction using Neural NARX. Until now only infinity-steps ahead were allowed. Now the users can set any steps they want.
    - Polynomial and Fourier are supported for now. New basis functions will be added in next releases.
    - No need to pass the number of inputs anymore.
    - Improved docstring.
    - Fixed minor grammatical and spelling mistakes.
    - many under the hood changes.

- Major: Support for old methods removed.
    - Now the old sysidentpy.PolynomialNarmax is not available anymore. All the old features are included in the new API with a lot of new features and performance improvements.

- API Change (new): sysidentpy.general_estimators.ModelPrediction
    - ModelPrediction class was adapted to support General Estimators as a stand-alone class.
    - predict: base method for prediction. Support infinity_steps ahead, one-step ahead and n-steps ahead prediction and any basis function.
    - _one_step_ahead_prediction: Perform the 1-step-ahead prediction for any basis function.
    - _n_step_ahead_prediction: Perform the n-step-ahead prediction for polynomial basis.
    - _model_prediction: Perform the infinity-step-ahead prediction for polynomial basis.
    - _narmax_predict: wrapper for NARMAX and NAR models.
    - _nfir_predict: wrapper for NFIR models.
    - _basis_function_predict: Perform the infinity-step-ahead prediction for basis functions other than polynomial.
    - basis_function_n_step_prediction: Perform the n-step-ahead prediction for basis functions other than polynomial.

- API Change (new): sysidentpy.neural_network.ModelPrediction
    - ModelPrediction class was adapted to support Neural NARX as a stand-alone class.
    - predict: base method for prediction. Support infinity_steps ahead, one-step ahead and n-steps ahead prediction and any basis function.
    - _one_step_ahead_prediction: Perform the 1-step-ahead prediction for any basis function.
    - _n_step_ahead_prediction: Perform the n-step-ahead prediction for polynomial basis.
    - _model_prediction: Perform the infinity-step-ahead prediction for polynomial basis.
    - _narmax_predict: wrapper for NARMAX and NAR models.
    - _nfir_predict: wrapper for NFIR models.
    - _basis_function_predict: Perform the infinity-step-ahead prediction for basis functions other than polynomial.
    - basis_function_n_step_prediction: Perform the n-step-ahead prediction for basis functions other than polynomial.

- API Change: Fit method for Neural NARX revamped.
    - No need to convert the data to tensor before calling Fit method anymore.

API Change: Keyword and positional arguments
    - Now users have to provide parameters with their names, as keyword arguments, instead of positional arguments. This is valid for every model class now.

- API Change (new): sysidentpy.utils.narmax_tools
    - New functions to help user getting useful information to build model. Now we have the regressor_code helper function to help to build neural NARX models.

- DOC: Improved Basic Steps notebook with new details about the prediction function.
- DOC: NARX Neural Network notebook was updated following the new api and showing new features.
- DOC: General Estimators notebook was updated following the new api and showing new features.
- DOC: Fixed minor grammatical and spelling mistakes, including Issues #77 and #78.
- DOC: Fix issues related to html on Jupyter notebooks examples on documentation.


## v0.1.9

### CONTRIBUTORS

- wilsonrljr
- samirmartins

### CHANGES

- The update **v0.1.9**  has been released with additional feature, minor API changes and fixes of the new features added in v0.1.7.

- MAJOR: Entropic Regression Algorithm
    - Added the new class ER to build NARX models using the Entropic Regression algorithm.
    - Only the Mutual Information KNN is implemented in this version and it may take too long to run on a high number of regressor, so the user should be careful regarding the number of candidates to put in the model.

- API: save_load
    - Added a function to save and load models from file.

- API: Added tests for python 3.9

- Fix : Change condition for n_info_values in FROLS. Now the value defined by the user is compared against X matrix shape instead of regressor space shape. This fix the Fourier basis function usage with more the 15 regressors in FROLS.

- DOC: Save and Load models
    - Added a notebook showing how to use the save_load method.

- DOC: Entropic Regression example
    - Added notebook with a simple example of how to use AOLS

- DOC: Fourier Basis Function Example
    - Added notebook with a simple example of how to use Fourier Basis Function

- DOC: PV forecasting benchmark
    - FIX AOLS prediction. The example was using the meta_mss model in prediction, so the results for AOLS were wrong.

- DOC: Fixed minor grammatical and spelling mistakes.

- DOC: Fix issues related to html on Jupyter notebooks examples on documentation.


## v0.1.8

### CONTRIBUTORS

- wilsonrljr

### CHANGES

- The update **v0.1.8**  has been released with additional feature, minor API changes and fixes of the new features added in v0.1.7.

- MAJOR: Ensemble Basis Functions
    - Now you can use different basis function together. For now we allow to use Fourier combined with Polynomial of different degrees.

- API change: Add "ensemble" parameter in basis function to combine the features of different basis function.

- Fix: N-steps ahead prediction for model_type="NAR" is working properly now with different forecast horizon.

- DOC: Air passenger benchmark
    - Remove unused code.
    - Use default hyperparameter in SysIdentPy models.

- DOC: Load forecasting benchmark
    - Remove unused code.
    - Use default hyperparameter in SysIdentPy models.

- DOC: PV forecasting benchmark
    - Remove unused code.
    - Use default hyperparameter in SysIdentPy models.


## v0.1.7

### CONTRIBUTORS

- wilsonrljr

### CHANGES

- The update **v0.1.7**  has been released with major changes and additional features. There are several API modifications and you will need to change your code to have the new (and upcoming) features. All modifications are meant to make future expansion easier.

- On the user's side, the changes are not that disruptive, but in the background there are many changes that allowed the inclusion of new features and bug fixes that would be complex to solve without the changes. Check the `documentation page <http://sysidentpy.org/notebooks.html>`__

- Many classes were basically rebuild it from scratch, so I suggest to look at the new examples of how to use the new version.

- I will present the main updates below in order to highlight features and usability and then all API changes will be reported.

- MAJOR: NARX models with Fourier basis function `Issue63 <https://github.com/wilsonrljr/sysidentpy/issues/63>`__, `Issue64 <https://github.com/wilsonrljr/sysidentpy/issues/64>`__
    - The user can choose which basis they want by importing it from sysidentpy.basis_function. Check the notebooks with examples of how to use it.
    - Polynomial and Fourier are supported for now. New basis functions will be added in next releases.

- MAJOR: NAR models `Issue58 <https://github.com/wilsonrljr/sysidentpy/issues/58>`__
    - It was already possible to build Polynomial NAR models, but with some hacks. Now the user just need to pass model_type="NAR" to build NAR models.
    - The user doesn't need to pass a vector of zeros as input anymore.
    - Works for any model structure selection algorithm (FROLS, AOLS, MetaMSS)

- Major: NFIR models `Issue59 <https://github.com/wilsonrljr/sysidentpy/issues/59>`__
    - NFIR models are models where the output depends only on past inputs. It was already possible to build Polynomial NFIR models, but with a lot of code on the user's side (much more than NAR, btw). Now the user just need to pass model_type="NFIR" to build NFIR models.
    - Works for any model structure selection algorithm (FROLS, AOLS, MetaMSS)

- Major: Select the order for the residues lags to use in Extended Least Squares - elag
    - The user can select the maximum lag of the residues to be used in the Extended Least Squares algorithm. In previous versions sysidentpy used a predefined subset of residual lags.
    - The degree of the lags follows the degree of the basis function

- Major: Residual analysis methods `Issue60 <https://github.com/wilsonrljr/sysidentpy/issues/60>`__
    - There are now specific functions to calculate the autocorrelation of the residuals and cross-correlation for the analysis of the residuals. In previous versions the calculation was limited to just two inputs, for example, limiting user usability.

- Major: Plotting methods `Issue61 <https://github.com/wilsonrljr/sysidentpy/issues/61>`__
    - The plotting functions are now separated from the models objects, so there are more flexibility regarding what to plot.
    - Residual plots were separated from the forecast plot

- API Change: sysidentpy.polynomial_basis.PolynomialNarmax is deprecated. Use sysidentpy.model_structure_selection.FROLS instead. `Issue64 <https://github.com/wilsonrljr/sysidentpy/issues/62>`__
    - Now the user doesn't need to pass the number of inputs as a parameter.
    - Added the elag parameter for unbiased_estimator. Now the user can define the number of lags of the residues for parameter estimation using the Extended Least Squares algorithm.
    - model_type parameter: now the user can select the model type to be built. The options are "NARMAX", "NAR" and "NFIR". "NARMAX" is the default. If you want to build a NAR model without any "hack", just set model_type="NAR". The same for "NFIR" models.

- API Change: sysidentpy.polynomial_basis.MetaMSS is deprecated. Use sysidentpy.model_structure_selection.MetaMSS instead. `Issue64 <https://github.com/wilsonrljr/sysidentpy/issues/64>`__
    - Now the user doesn't need to pass the number of inputs as a parameter.
    - Added the elag parameter for unbiased_estimator. Now the user can define the number of lags of the residues for parameter estimation using the Extended Least Squares algorithm.

- API Change: sysidentpy.polynomial_basis.AOLS is deprecated. Use sysidentpy.model_structure_selection.AOLS instead. `Issue64 <https://github.com/wilsonrljr/sysidentpy/issues/64>`__

- API Change: sysidentpy.polynomial_basis.SimulatePolynomialNarmax is deprecated. Use sysidentpy.simulation.SimulateNARMAX instead.

- API Change: Introducing sysidentpy.basis_function. Because NARMAX models can be built on different basis function, a new module is added to make easier to implement new basis functions in future updates `Issue64 <https://github.com/wilsonrljr/sysidentpy/issues/64>`__.
    - Each basis function class must have a fit and predict method to be used in training and prediction respectively.

- API Change: unbiased_estimator method moved to Estimators class.
    - added elag option
    - change the build_information_matrix method to build_output_matrix

- API Change (new): sysidentpy.narmax_base
    - This is the new base for building NARMAX models. The classes have been rewritten to make it easier to expand functionality.

- API Change (new): sysidentpy.narmax_base.GenerateRegressors
    - create_narmax_code: Creates the base coding that allows representation for the NARMAX, NAR, and NFIR models.
    - regressor_space: Creates the encoding representation for the NARMAX, NAR, and NFIR models.

- API Change (new): sysidentpy.narmax_base.ModelInformation
    - _get_index_from_regressor_code: Get the index of the model code representation in regressor space.
    - _list_output_regressor_code: Create a flattened array of output regressors.
    - _list_input_regressor_code: Create a flattened array of input regressors.
    - _get_lag_from_regressor_code: Get the maximum lag from array of regressors.
    - _get_max_lag_from_model_code: the name says it all.
    - _get_max_lag: Get the maximum lag from ylag and xlag.

- API Change (new): sysidentpy.narmax_base.InformationMatrix
    - _create_lagged_X: Create a lagged matrix of inputs without combinations.
    - _create_lagged_y: Create a lagged matrix of the output without combinations.
    - build_output_matrix: Build the information matrix of output values.
    - build_input_matrix: Build the information matrix of input values.
    - build_input_output_matrix: Build the information matrix of input and output values.

- API Change (new): sysidentpy.narmax_base.ModelPrediction
    - predict: base method for prediction. Support infinity_steps ahead, one-step ahead and n-steps ahead prediction and any basis function.
    - _one_step_ahead_prediction: Perform the 1-step-ahead prediction for any basis function.
    - _n_step_ahead_prediction: Perform the n-step-ahead prediction for polynomial basis.
    - _model_prediction: Perform the infinity-step-ahead prediction for polynomial basis.
    - _narmax_predict: wrapper for NARMAX and NAR models.
    - _nfir_predict: wrapper for NFIR models.
    - _basis_function_predict: Perform the infinity-step-ahead prediction for basis functions other than polynomial.
    - basis_function_n_step_prediction: Perform the n-step-ahead prediction for basis functions other than polynomial.

- API Change (new): sysidentpy.model_structure_selection.FROLS `Issue62 <https://github.com/wilsonrljr/sysidentpy/issues/62>`__, `Issue64 <https://github.com/wilsonrljr/sysidentpy/issues/64>`__
    - Based on the old sysidentpy.polynomial_basis.PolynomialNARMAX. The class has been rebuilt with new functions and optimized code.
    - Enforcing keyword-only arguments. This is an effort to promote clear and non-ambiguous use of the library.
    - Add support for new basis functions.
    - The user can choose the residual lags.
    - No need to pass the number of inputs anymore.
    - Improved docstring.
    - Fixed minor grammatical and spelling mistakes.
    - New prediction method.
    - many under the hood changes.

- API Change (new): sysidentpy.model_structure_selection.MetaMSS `Issue64 <https://github.com/wilsonrljr/sysidentpy/issues/64>`__
    - Based on the old sysidentpy.polynomial_basis.MetaMSS. The class has been rebuilt with new functions and optimized code.
    - Enforcing keyword-only arguments. This is an effort to promote clear and non-ambiguous use of the library.
    - The user can choose the residual lags.
    - Extended Least Squares support.
    - Add support for new basis functions.
    - No need to pass the number of inputs anymore.
    - Improved docstring.
    - Fixed minor grammatical and spelling mistakes.
    - New prediction method.
    - many under the hood changes.

- API Change (new): sysidentpy.model_structure_selection.AOLS `Issue64 <https://github.com/wilsonrljr/sysidentpy/issues/64>`__
    - Based on the old sysidentpy.polynomial_basis.AOLS. The class has been rebuilt with new functions and optimized code.
    - Enforcing keyword-only arguments. This is an effort to promote clear and non-ambiguous use of the library.
    - Add support for new basis functions.
    - No need to pass the number of inputs anymore.
    - Improved docstring.
    - Change "l" parameter to "L".
    - Fixed minor grammatical and spelling mistakes.
    - New prediction method.
    - many under the hood changes.

- API Change (new): sysidentpy.simulation.SimulateNARMAX
    - Based on the old sysidentpy.polynomial_basis.SimulatePolynomialNarmax. The class has been rebuilt with new functions and optimized code.
    - Fix the Extended Least Squares support.
    - Fix n-steps ahead prediction and 1-step ahead prediction.
    - Enforcing keyword-only arguments. This is an effort to promote clear and non-ambiguous use of the library.
    - The user can choose the residual lags.
    - Improved docstring.
    - Fixed minor grammatical and spelling mistakes.
    - New prediction method.
    - Do not inherit from the structure selection algorithm anymore, only from narmax_base. Avoid circular import and other issues.
    - many under the hood changes.

- API Change (new): sysidentpy.residues
    - compute_residues_autocorrelation: the name says it all.
    - calculate_residues: get the residues from y and yhat.
    - get_unnormalized_e_acf: compute the unnormalized autocorrelation of the residues.
    - compute_cross_correlation: compute cross correlation between two signals.
    - _input_ccf
    - _normalized_correlation: compute the normalized correlation between two signals.

- API Change (new): sysidentpy.utils.plotting
    - plot_results: plot the forecast
    - plot_residues_correlation: the name says it all.

- API Change (new): sysidentpy.utils.display_results
    - results: return the model regressors, estimated parameter and ERR index of the fitted model in a table.

- DOC: Air passenger benchmark `Issue65 <https://github.com/wilsonrljr/sysidentpy/issues/65>`__
    - Added notebook with Air passenger forecasting benchmark.
    - We compare SysIdentPy against prophet, neuralprophet, autoarima, tbats and many more.

- DOC: Load forecasting benchmark `Issue65 <https://github.com/wilsonrljr/sysidentpy/issues/65>`__
    - Added notebook with load forecasting benchmark.

- DOC: PV forecasting benchmark `Issue65 <https://github.com/wilsonrljr/sysidentpy/issues/65>`__
    - Added notebook with PV forecasting benchmark.

- DOC: Presenting main functionality
    - Example rewritten following the new api.
    - Fixed minor grammatical and spelling mistakes.

- DOC: Multiple Inputs usage
    - Example rewritten following the new api
    - Fixed minor grammatical and spelling mistakes.

- DOC: Information Criteria - Examples
    - Example rewritten following the new api.
    - Fixed minor grammatical and spelling mistakes.

- DOC: Important notes and examples of how to use Extended Least Squares
    - Example rewritten following the new api.
    - Fixed minor grammatical and spelling mistakes.

- DOC: Setting specific lags
    - Example rewritten following the new api.
    - Fixed minor grammatical and spelling mistakes.

- DOC: Parameter Estimation
    - Example rewritten following the new api.
    - Fixed minor grammatical and spelling mistakes.

- DOC: Using the Meta-Model Structure Selection (MetaMSS) algorithm for building Polynomial NARX models
    - Example rewritten following the new api.
    - Fixed minor grammatical and spelling mistakes.

- DOC: Using the Accelerated Orthogonal Least-Squares algorithm for building Polynomial NARX models
    - Example rewritten following the new api.
    - Fixed minor grammatical and spelling mistakes.

- DOC: Example: F-16 Ground Vibration Test benchmark
    - Example rewritten following the new api.
    - Fixed minor grammatical and spelling mistakes.

- DOC: Building NARX Neural Network using Sysidentpy
    - Example rewritten following the new api.
    - Fixed minor grammatical and spelling mistakes.

- DOC: Building NARX models using general estimators
    - Example rewritten following the new api.
    - Fixed minor grammatical and spelling mistakes.

- DOC: Simulate a Predefined Model
    - Example rewritten following the new api.
    - Fixed minor grammatical and spelling mistakes.

- DOC: System Identification Using Adaptive Filters
    - Example rewritten following the new api.
    - Fixed minor grammatical and spelling mistakes.

- DOC: Identification of an electromechanical system
    - Example rewritten following the new api.
    - Fixed minor grammatical and spelling mistakes.

- DOC: Example: N-steps-ahead prediction - F-16 Ground Vibration Test benchmark
    - Example rewritten following the new api.
    - Fixed minor grammatical and spelling mistakes.

- DOC: Introduction to NARMAX models
    - Fixed grammatical and spelling mistakes.



## v0.1.6

### CONTRIBUTORS

- wilsonrljr

### CHANGES

- MAJOR: Meta-Model Structure Selection Algorithm (Meta-MSS).
    - A new method for build NARMAX models based on metaheuristics. The algorithm uses a Binary hybrid Particle Swarm Optimization and Gravitational Search Algorithm with a new cost function to build parsimonious models.
    - New class for the BPSOGSA algorithm. New algorithms can be adapted in the Meta-MSS framework.
	- Future updates will add NARX models for classification and multiobjective model structure selection.

- MAJOR: Accelerated Orthogonal Least-Squares algorithm.
    - Added the new class AOLS to build NARX models using the Accelerated Orthogonal Least-Squares algorithm.
    - At the best of my knowledge, this is the first time this algorithm is used in the NARMAX framework. The tests I've made are promising, but use it with caution until the results are formalized into a research paper.

- Added notebook with a simple example of how to use MetaMSS and a simple model comparison of the Electromechanical system.

- Added notebook with a simple example of how to use AOLS

- Added ModelInformation class. This class have methods to return model information such as max_lag of a model code.
    - added _list_output_regressor_code
    - added _list_input_regressor_code
    - added _get_lag_from_regressor_code
    - added _get_max_lag_from_model_code

- Minor performance improvement: added the argument "predefined_regressors" in build_information_matrix function on base.py to improve the performance of the Simulation method.

- Pytorch is now an optional dependency. Use pip install sysidentpy['full']

- Fix code format issues.

- Fixed minor grammatical and spelling mistakes.

- Fix issues related to html on Jupyter notebooks examples on documentation.

- Updated Readme with examples of how to use.

- Improved descriptions and comments in methods.

- metaheuristics.bpsogsa (detailed description on code docstring)
    - added evaluate_objective_function
    - added optimize
    - added generate_random_population
    - added mass_calculation
    - added calculate_gravitational_constant
    - added calculate_acceleration
    - added update_velocity_position

- FIX issue #52


## v0.1.5

### CONTRIBUTORS

- wilsonrljr

### CHANGES

- MAJOR: n-steps-ahead prediction.
    - Now you can define the numbers of steps ahead in the predict function.
	- Only for Polynomial models for now. Next update will bring this functionality to Neural NARX and General Estimators.

- MAJOR: Simulating predefined models.
    - Added the new class SimulatePolynomialNarmax to handle the simulation of known model structures.
    - Now you can simulate predefined models by just passing the model structure codification. Check the notebook examples.

- Added 4 new notebooks in the example section.

- Added iterative notebooks. Now you can run the notebooks in Jupyter notebook section of the documentation in Colab.

- Fix code format issues.

- Added new tests for SimulatePolynomialNarmax and generate_data.

- Started changes related to numpy 1.19.4 update. There are still some Deprecation warnings that will be fixed in next update.

- Fix issues related to html on Jupyter notebooks examples on documentation.

- Updated Readme with examples of how to use.


## v0.1.4

### CONTRIBUTORS

- wilsonrljr

### CHANGES

- MAJOR: Introducing NARX Neural Network in SysIdentPy.
    - Now you can build NARX Neural Network on SysIdentPy.
    - This feature is built on top of Pytorch. See the docs for more details and examples of how to use.

- MAJOR: Introducing general estimators in SysIdentPy.
    - Now you are able to use any estimator that have Fit/Predict methods (estimators from Sklearn and Catboost, for example) and build NARX models based on those estimators.
    - We use the core functions of SysIdentPy and keep the Fit/Predict approach from those estimators to keep the process easy to use.
    - More estimators are coming soon like XGboost.

- Added notebooks to show how to build NARX neural Network.

- Added notebooks to show how to build NARX models using general estimators.

- Changed the default parameters of the plot_results function.

- NOTE: We will keeping improving the Polynomial NARX models (new model structure selection algorithms and multiobjective identification
is on our roadmap). These recent modifications will allow us to introduce new NARX models like PWARX models very soon.

- New template for the documentation site.

- Fix issues related to html on Jupyter notebooks examples on documentation.

- Updated Readme with examples of how to use.

## v0.1.3

### CONTRIBUTORS

- wilsonrljr
- renard162

### CHANGES

- Fixed a bug concerning the xlag and ylag in multiple input scenarios.
- Refactored predict function. Improved performance up to 87% depending on the number of regressors.
- You can set lags with different size for each input.
- Added a new function to get the max value of xlag and ylag. Work with int, list, nested lists.
- Fixed tests for information criteria.
- Added SysIdentPy logo.
- Refactored code of all classes following PEP 8 guidelines to improve readability.
- Added Citation information on Readme.
- Changes on information Criteria tests.
- Added workflow to run the tests when merge branch into master.
- Added new site domain.
- Updated docs.
