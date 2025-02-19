---
template: overrides/main.html
title: Getting Started
---



With **SysIdentPy**, you can:

- Build and customize nonlinear forecasting models.
- Utilize state-of-the-art techniques for model structure selection and parameter estimation.
- Experiment with neural NARX models and other advanced algorithms.

Check our [documentation](https://sysidentpy.org)!

For an in depth documentation, check our companion book:

<a href="https://sysidentpy.org/book/0%20-%20Preface/">
  <img src="https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/Nonlinear_System_identification.png?raw=true" alt="Nonlinear System Identification" style="width: 200px; height: auto;" />
</a>



### Requirements

`SysIdentPy` requires:

- Python (>= 3.7)
- NumPy (>= 1.9.2) for numerical algorithms
- Matplotlib >= 3.3.2 for static plotting and visualizations
- Pytorch (>=1.7.1) for building NARX neural networks
- scipy (>= 1.7.0) for numerical and optimization algorithms

The library is compatible with Linux, Windows, and macOS. Some examples may also require additional packages like pandas.

For more details, check our [installation guide](https://sysidentpy.org/landing-page/getting-started/)

## What are the main features of SysIdentPy?

| Feature | What is this? |
|----------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| NARMAX philosophy | You can build variations of NARMAX models like NARX, NAR, NARMA, NFIR, ARMA, ARX, AR, and others. |
| Model Structure Selection | Easy-to-use methods to select the best terms to build your models, including FROLS and MetaMSS and several combinations with parameter estimation techniques to select the model terms. |
| Basis Function | You can use up to 8 different basis functions to build your models. You can set linear and nonlinear basis functions and ensemble them to get custom NARMAX models. |
| Parameter Estimation | More than 15 methods to estimate the model parameters and test different structure selection scenarios. |
| Multiobjective Parameter Estimation | You can use affine information to estimate the model parameters minimizing different objective functions. |
| Model Simulation | You can reproduce results from papers easily with SimulateNARMAX class. Moreover, you can test published models with different parameter estimation methods and compare the performance. |
| Neural NARX | You can use SysIdentPy with Pytorch to create custom neural NARX models architectures which support all the optimizers and loss functions from Pytorch. |
| General Estimators | You can use estimators from packages like scikit-learn, Catboost, and many other compatible interfaces and composition tools to create NARMAX models. |

## Why does SysIdentPy exist?

SysIdentPy aims to be a free and open-source package to help the community to design NARMAX models for System Identification and TimeSeries Forecasting. More than that, be a free and robust alternative to one of the most used tools to build NARMAX models, which is the Matlab's System Identification Toolbox.

The project is actively maintained by Wilson R. L. Junior and looking for contributors.

## How do I use SysIdentPy?

The [SysIdentPy documentation](https://sysidentpy.org) includes more than 20 examples to help get you started:
- Typical "Hello World" example, for an [entry-level description of the main SysIdentPy concepts](https://sysidentpy.org/examples/basic_steps/)
- A dedicated section focusing on SysIdentPy features, like model structure selection algorithms, basis functions, parameter estimation, and more.
- A dedicated section focusing on use cases using SysIdentPy with real world datasets. Besides, there is some brief comparisons and benchmarks against other time series tools, like Prophet, Neural Prophet, ARIMA, and more.







# Getting Started

Welcome to SysIdentPy's documentation! Learn how to get started with SysIdentPy in your project. Then, explore SysIdentPy's main concepts and discover additional resources to help you model dynamic systems and time series.

> **Looking for more details on NARMAX models?**
> For comprehensive information on models, methods, and a wide range of examples and benchmarks implemented in SysIdentPy, check out our book:
> [*Nonlinear System Identification and Forecasting: Theory and Practice With SysIdentPy*](https://sysidentpy.org/book/0%20-%20Preface/)
>
> This book provides in-depth guidance to support your work with SysIdentPy.
>
> You can also explore the [tutorials in the documentation](https://sysidentpy.org/examples/basic_steps/) for practical, hands-on examples.

## What is SysIdentPy

SysIdentPy is an open-source Python module for System Identification using **NARMAX** models built on top of **numpy** and is distributed under the 3-Clause BSD license. SysIdentPy provides an easy-to-use and  flexible framework for building Dynamical Nonlinear Models for time series and dynamic systems.

## Installation

SysIdentPy is published as a [Python package] and can be installed with
`pip`, ideally by using a [virtual environment]. If not, scroll down and expand
the help box. Install with:

<div class="custom-card-container">

<div class="custom-card">
    <h3>📦 Install SysIdentPy</h3>
    <p>Choose the installation option that fits your needs:</p>

    <div class="tab-container">
        <input type="radio" id="tab-latest" name="tab-group" checked>
        <label for="tab-latest">Latest</label>
        <div class="tab-content">
            <pre><code>pip install sysidentpy</code></pre>
        </div>

        <input type="radio" id="tab-neural" name="tab-group">
        <label for="tab-neural">Neural NARX Support</label>
        <div class="tab-content">
            <pre><code>pip install sysidentpy["all"]</code></pre>
        </div>

        <input type="radio" id="tab-version" name="tab-group">
        <label for="tab-version">Version x.y.z</label>
        <div class="tab-content">
            <pre><code>pip install sysidentpy=="0.5.3"</code></pre>
        </div>
    </div>
</div>

</div>



??? question "How to manage my projects dependencies?"

    If you don't have prior experience with Python, we recommend reading [Using Python's pip to Manage Your Projects' Dependencies], which is a really good introduction on the mechanics of Python package management and helps you troubleshoot if you run into errors.

  [Python package]: https://pypi.org/project/sysidentpy/
  [virtual environment]: https://realpython.com/what-is-pip/#using-pip-in-a-python-virtual-environment
  [Using Python's pip to Manage Your Projects' Dependencies]: https://realpython.com/what-is-pip/


### with git

SysIdentPy can be used directly from [GitHub] by cloning the
repository into a subfolder of your project root which might be useful if you
want to use the very latest version:

```
git clone https://github.com/wilsonrljr/sysidentpy.git
```

  [GitHub]: https://github.com/wilsonrljr/sysidentpy


## Main Concepts



## Do you like **SysIdentPy**?

Would you like to help SysIdentPy, other users, and the author? You can "star" SysIdentPy in GitHub by clicking in the star button at the top right of the page: <a href="https://github.com/wilsonrljr/sysidentpy" class="external-link" target="_blank">https://github.com/wilsonrljr/sysidentpy</a>. ⭐️

Starring a repository makes it easy to find it later and help you to find similar projects on GitHub based on Github recommendation contents. Besides, by starring a repository also shows appreciation to the SysIdentPy maintainer for their work.

[:octicons-star-fill-24:{ .mdx-heart } &nbsp; Join our <span class="mdx-sponsorship-count" data-mdx-component="sponsorship-count"></span> "Star" in github][wilsonrljr's sponsor profile]{ .md-button .md-button--primary .mdx-sponsorship-button }

  [wilsonrljr's sponsor profile]: https://github.com/sponsors/wilsonrljr

