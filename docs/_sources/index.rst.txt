.. sys-identpy documentation master file, created by
   sphinx-quickstart on Sun Mar 15 08:37:08 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to SysIdentPy's documentation!
======================================

**SysIdentPy** is a Python module for System Identification using **NARMAX** models built on top of **numpy** and is distributed under the 3-Clause BSD license.

The project was started in by Wilson R. L. Junior, Luan Pascoal C. Andrade and Samir A. M. Martins as a project for System Identification discipline. Samuel joined early in 2019 and since then have contributed.

.. seealso::
	The examples directory has several Jupyter notebooks presenting basic tutorials of how to use the package and some specific applications of **SysIdentPy**. `Try it out! <http://sysidentpy.org/notebooks.html>`__


Changelog
---------

See the `changelog <https://github.com/wilsonrljr/sysidentpy.whats_is_new.md>`__
for a history of notable changes to **SysIdentPy**.


Development
-----------

We welcome new contributors of all experience levels. The **SysIdentPy** community goals are to be helpful, welcoming, and effective.

.. note::
	We use the `pytest` package for testing. The test functions are located in tests subdirectories at each folder inside **SysIdentPy**, which check the validity of the algorithms.

Run the `pytest` in the respective folder to perform all the tests of the corresponding sub-packages.

Currently, we have around 81% of code coverage.

You can install pytest using ::

	pip install -U pytest

Example of how to run the tests:
--------------------------------

Open a terminal emulator of your choice and go to a subdirectory, e.g, ::

	\sysidentpy\metrics\

Just type :code:`pytest` and you get a result like ::


	========== test session starts ==========

	platform linux -- Python 3.7.6, pytest-5.4.2, py-1.8.1, pluggy-0.13.1

	rootdir: ~/sysidentpy

	plugins: cov-2.8.1

	collected 12 items

	tests/test_regression.py ............ [100%]

	========== 12 passed in 2.45s ==================

You can also see the code coverage using the :code:`pytest-cov` package. First, install :code:`pytest-cov` using ::

	pip install pytest-cov

Run the command below in the **SysIdentPy** root directory, to generate the report. ::

	pytest --cov=.


Source code
-----------

You can check the latest sources with the command::

    git clone https://github.com/wilsonrljr/sysidentpy.git

Project History
---------------

The project was started by Wilson R. L. Junior, Luan Pascoal and Samir A. M. Martins as a project for System Identification discipline. Samuel joined early in 2019 and since then have contributed.

The initial purpose was to learn the python language. Over time, the project has matured to the state it is in today.

The project is currently maintained by its creators and looking for
contributors.

Communication
-------------

- Discord server: https://discord.gg/8eGE3PQ
- Website(soon): http://sysidentpy.org

Citation
--------

If you use **SysIdentPy** on your project, please `drop me a line <wilsonrljr@outlook.com>`__.

If you use **SysIdentPy** on your scientific publication, we would appreciate citations to the following paper:

- Lacerda et al., (2020). SysIdentPy: A Python package for System Identification using NARMAX models. Journal of Open Source Software, 5(54), 2384, https://doi.org/10.21105/joss.02384 ::

	@article{Lacerda2020,
	  doi = {10.21105/joss.02384},
	  url = {https://doi.org/10.21105/joss.02384},
	  year = {2020},
	  publisher = {The Open Journal},
	  volume = {5},
	  number = {54},
	  pages = {2384},
	  author = {Wilson Rocha Lacerda Junior and Luan Pascoal Costa da Andrade and Samuel Carlos Pessoa Oliveira and Samir Angelo Milani Martins},
	  title = {SysIdentPy: A Python package for System Identification using NARMAX models},
	  journal = {Journal of Open Source Software}
	}

Inspiration
-----------

The documentation and structure (even this section) is openly inspired by sklearn, einsteinpy, and many others as we used (and keep using) them to learn.

Contents
--------

.. toctree::
    :maxdepth: 1

    installation
    introduction_to_narmax
    user_guide
    dev_guide
    notebooks
    changelog/v0.1.3
    code
