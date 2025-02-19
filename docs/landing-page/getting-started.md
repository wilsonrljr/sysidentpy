---
template: overrides/main.html
title: Getting Started
---

# Getting Started

Welcome to SysIdentPy's documentation! Learn how to get started with SysIdentPy in your project. Then, explore SysIdentPy's main concepts and discover additional resources to help you model dynamic systems and time series.

<div class="custom-collapsible-card">
  <input type="checkbox" id="toggle-info">
  <label for="toggle-info">
    📚 <strong>Looking for more details on NARMAX models?</strong>
    <span class="arrow">▼</span>
  </label>
  <div class="collapsible-content">
    <p>
      For comprehensive information on models, methods, and a wide range of examples and benchmarks implemented in <strong>SysIdentPy</strong>, check out our book:
    </p>
    <a href="https://sysidentpy.org/book/0%20-%20Preface/" target="_blank">
      <em><strong>Nonlinear System Identification and Forecasting: Theory and Practice With SysIdentPy</strong></em>
    </a>
    <p>
      This book provides in-depth guidance to support your work with <strong>SysIdentPy</strong>.
    </p>
    <p>
      🛠️ You can also explore the <a href="https://sysidentpy.org/examples/basic_steps/" target="_blank"><strong>tutorials in the documentation</strong></a> for practical, hands-on examples.
    </p>
  </div>
</div>

## What is SysIdentPy

SysIdentPy is an open-source Python module for System Identification using **NARMAX** models built on top of **numpy** and is distributed under the 3-Clause BSD license. SysIdentPy provides an easy-to-use and  flexible framework for building Dynamical Nonlinear Models for time series and dynamic systems.

With **SysIdentPy**, you can:

- Build and customize nonlinear forecasting models.
- Utilize state-of-the-art techniques for model structure selection and parameter estimation.
- Experiment with neural NARX models and other advanced algorithms.

## Installation

SysIdentPy is published as a [Python package] and can be installed with
`pip`, ideally by using a [virtual environment]. If not, scroll down and expand
the help box. Install with:

<div class="custom-card">
  <div class="tab-container">
    <!-- Latest Tab -->
    <input type="radio" id="tab-latest" name="tab-group" checked>
    <label for="tab-latest">Latest</label>
    <div class="tab-content">
      <pre><code>pip install sysidentpy</code></pre>
    </div>

    <!-- Neural NARX Support Tab -->
    <input type="radio" id="tab-neural" name="tab-group">
    <label for="tab-neural">Neural NARX Support</label>
    <div class="tab-content">
      <pre><code>pip install sysidentpy["all"]</code></pre>
    </div>

    <!-- Version x.y.z Tab -->
    <input type="radio" id="tab-version" name="tab-group">
    <label for="tab-version">Specific Version</label>
    <div class="tab-content">
      <pre><code>pip install sysidentpy=="0.5.3"</code></pre>
    </div>

    <!-- Nightly Builds -->
    <input type="radio" id="tab-git" name="tab-group">
    <label for="tab-git">From Git</label>
    <div class="tab-content">
      <pre><code>pip install git+https://github.com/wilsonrljr/sysidentpy.git</code></pre>
    </div>
  </div>
</div>


<div class="custom-collapsible-card">
  <input type="checkbox" id="toggle-dependencies">
  <label for="toggle-dependencies">
    ❓ <strong>How to manage my projects dependencies?</strong>
    <span class="arrow">▼</span>
  </label>
  <div class="collapsible-content">
    <p>
      If you don't have prior experience with Python, we recommend reading
      <a href="https://pip.pypa.io/en/stable/user_guide/" target="_blank">
        Using Python's pip to Manage Your Projects' Dependencies
      </a>, which is a really good introduction on the mechanics of Python package management and helps you troubleshoot if you run into errors.
    </p>
  </div>
</div>


  [Python package]: https://pypi.org/project/sysidentpy/
  [virtual environment]: https://realpython.com/what-is-pip/#using-pip-in-a-python-virtual-environment
  [Using Python's pip to Manage Your Projects' Dependencies]: https://realpython.com/what-is-pip/


## What are the main features of SysIdentPy?

<div class="feature-grid">
  <div class="feature-card">
    <h3>🧩 NARMAX Philosophy</h3>
    <p>Build variations like <strong>NARX</strong>, <strong>NAR</strong>, <strong>ARMA</strong>, <strong>NFIR</strong>, and more.</p>
  </div>
  <div class="feature-card">
    <h3>📝 Model Structure Selection</h3>
    <p>Use methods like <strong>FROLS</strong>, <strong>MetaMSS</strong>, and combinations with parameter estimation techniques.</p>
  </div>
  <div class="feature-card">
    <h3>🔗 Basis Function</h3>
    <p>Choose from <strong>8+ basis functions</strong>, combining linear and nonlinear types for custom NARMAX models.</p>
  </div>
  <div class="feature-card">
    <h3>🎯 Parameter Estimation</h3>
    <p>Over <strong>15 parameter estimation methods</strong> for exploring various structure selection scenarios.</p>
  </div>
  <div class="feature-card">
    <h3>⚖️ Multiobjective Estimation</h3>
    <p>Minimize different objective functions using <strong>affine information</strong> for parameter estimation.</p>
  </div>
  <div class="feature-card">
    <h3>🔄 Model Simulation</h3>
    <p>Reproduce paper results easily with <strong>SimulateNARMAX</strong>. Test and compare published models effortlessly.</p>
  </div>
  <div class="feature-card">
    <h3>🤖 Neural NARX (PyTorch)</h3>
    <p>Integrate with <strong>PyTorch</strong> for custom neural NARX architectures using all PyTorch optimizers and loss functions.</p>
  </div>
  <div class="feature-card">
    <h3>🛠️ General Estimators</h3>
    <p>Compatible with <strong>scikit-learn</strong>, <strong>Catboost</strong>, and more for creating NARMAX models.</p>
  </div>
</div>


## How do I use SysIdentPy?

The [SysIdentPy documentation](https://sysidentpy.org) includes more than 20 examples to help get you started:
- Typical "Hello World" example, for an [entry-level description of the main SysIdentPy concepts](https://sysidentpy.org/examples/basic_steps/)
- A dedicated section focusing on SysIdentPy features, like model structure selection algorithms, basis functions, parameter estimation, and more.
- A dedicated section focusing on use cases using SysIdentPy with real world datasets. Besides, there is some brief comparisons and benchmarks against other time series tools, like Prophet, Neural Prophet, ARIMA, and more.


## Do you like **SysIdentPy**?

Would you like to help SysIdentPy, other users, and the author? You can "star" SysIdentPy in GitHub by clicking in the star button at the top right of the page: <a href="https://github.com/wilsonrljr/sysidentpy" class="external-link" target="_blank">https://github.com/wilsonrljr/sysidentpy</a>. ⭐️

Starring a repository makes it easy to find it later and help you to find similar projects on GitHub based on Github recommendation contents. Besides, by starring a repository also shows appreciation to the SysIdentPy maintainer for their work.

[:octicons-star-fill-24:{ .mdx-heart } &nbsp; Join our <span class="mdx-sponsorship-count" data-mdx-component="sponsorship-count"></span> "Star" in github][wilsonrljr's sponsor profile]{ .md-button .md-button--primary .mdx-sponsorship-button }

  [wilsonrljr's sponsor profile]: https://github.com/sponsors/wilsonrljr

