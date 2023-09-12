---
template: overrides/main.html
title: Getting Started
---

# Getting Started

SysIdentPy is a Python module for System Identification using NARMAX models built on top of numpy and is distributed under the 3-Clause BSD license.

## Do you like **SysIdentPy**?

Would you like to help SysIdentPy, other users, and the author? You can "star" SysIdentPy in GitHub by clicking in the star button at the top right of the page: <a href="https://github.com/wilsonrljr/sysidentpy" class="external-link" target="_blank">https://github.com/wilsonrljr/sysidentpy</a>. ⭐️

Starring a repository makes it easy to find it later and help you to find similar projects on GitHub based on Github recommendation contents. Besides, by starring a repository also shows appreciation to the SysIdentPy maintainer for their work.

[:octicons-star-fill-24:{ .mdx-heart } &nbsp; Join our <span class="mdx-sponsorship-count" data-mdx-component="sponsorship-count"></span> "Star" in github][wilsonrljr's sponsor profile]{ .md-button .md-button--primary .mdx-sponsorship-button }

  [wilsonrljr's sponsor profile]: https://github.com/sponsors/wilsonrljr


Requirements
------------

SysIdentPy requires:

| Dependency | version     | Comment                                              |
|------------|-------------|------------------------------------------------------|
| python     | >=3.7,<3.10 |                                                      |
| numpy      | >=1.9.2     | for all numerical algorithms                         |
| scipy      | >=1.7.0     | for some linear regression methods                   |
| matplotlib | >=3.3.2     | for static plotting and visualizations               |
| torch      | >=1.7.1     | Only necessary if you want to use Neural NARX models |


| Platform | Status |
|----------|--------|
| Windows  | ok     |
| Linux    | ok     |
| Mac OS   | ok     |

SysIdentPy **do not** to support Python 2.7.

A few examples require pandas >= 0.18.0. However, it is not required to use SysIdentPy.

## Installation

### with pip <small>recommended</small> { #with-pip data-toc-label="with pip" }

SysIdentPy is published as a [Python package] and can be installed with
`pip`, ideally by using a [virtual environment]. If not, scroll down and expand
the help box. Install with:

=== "Latest"

    ``` sh
    pip install sysidentpy
    ```

=== "Neural NARX Support"

    ``` sh
    pip install sysidentpy["all"]
    ```

=== "v0.1.6"

    ``` sh
    pip install sysidentpy=="0.1.6"
    ```

---

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