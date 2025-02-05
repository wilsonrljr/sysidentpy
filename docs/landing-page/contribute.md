---
template: overrides/main.html
title: Contribute
---

# Contributing

SysIdentPy is intended to be a community project, hence all contributions are welcome! There exist many possible use cases in System Identification field and we can not test all scenarios without your help! If you find any bugs or have suggestions, please report them on [issue tracker] on GitHub.

>We welcome new contributors of all experience levels. The SysIdentPy community goals are to be helpful, welcoming, and effective.

  [issue tracker]: https://github.com/wilsonrljr/sysidentpy/issues

## Help others with issues in GitHub

You can see <a href="https://github.com/wilsonrljr/sysidentpy/issues" class="external-link" target="_blank">existing issues</a> and try and help others, most of the times they are questions that you might already know the answer for.

## Watch the GitHub repository

You can [watch] SysIdentPy in GitHub (clicking the "watch" button at the top right):
  [watch]: https://github.com/wilsonrljr/sysidentpy

If you select "Watching" instead of "Releases only" you will receive notifications when someone creates a new issue.

Then you can try and help them solve those issues.

## Documentation

Documentation is as important as the library itself. English is not the primary language of the main authors, so if you find any typo or anything wrong do not hesitate to point out to us.

## Create a Pull Request

You can [contribute](contribute.md){.internal-link target=_blank} to the source code with Pull Requests, for example:

* To fix a typo you found on the documentation.
* To share an article, video, or podcast you created or found about SysIdentPy.
* To propose new documentation sections.
* To fix an existing issue/bug.
* To add a new feature.

## Development environment

These are some basic steps to help us with code:

- [x]	Install and Setup Git on your computer.
- [x]	[Fork] SysIdentPy.
- [x]	[Clone] the fork on your local machine.
- [x]	Create a new branch.
- [x]	Make changes following the coding style of the project (or suggesting improvements).
- [x]	Run the tests.
- [x]	Write and/or adapt existing test if needed.
- [x]	Add documentation if needed.
- [x]	Commit.
- [x]	[Push] to your fork.
- [x]	Open a [pull_request].

  [Fork]: https://help.github.com/articles/fork-a-repo/
  [Clone]: https://help.github.com/articles/cloning-a-repository/
  [Push]: https://help.github.com/articles/pushing-to-a-remote/
  [pull_request]: https://help.github.com/articles/creating-a-pull-request/


### Environment

Clone the repository using

```console
git clone https://github.com/wilsonrljr/sysidentpy.git
```

If you already cloned the repository and you know that you need to deep dive in the code, here are some guidelines to set up your environment.

#### Virtual environment with `venv`

You can create a virtual environment in a directory using Python's `venv` module or Conda:

=== "venv"

    ```console
    $ python -m venv env
    ```

=== "conda"

    ```console
    conda create -n env
    ```


That will create a directory `./env/` with the Python binaries and then you will be able to install packages for that isolated environment.

#### Activate the environment

If you created the environment using Python's `venv` module, activate it with:

=== "Linux, macOS"

    ```console
    source ./env/bin/activate
    ```

=== "Windows PowerShell"

    ```console
    .\env\Scripts\Activate.ps1
    ```

=== "Windows Bash"

    Or if you use Bash for Windows (e.g. <a href="https://gitforwindows.org/" class="external-link" target="_blank">Git Bash</a>):

    ```console
    source ./env/Scripts/activate
    ```

If you created the environment using Conda, activate it with:

=== "Conda Bash"

    Or if you use Bash for Windows (e.g. <a href="https://gitforwindows.org/" class="external-link" target="_blank">Git Bash</a>):

    ```console
    conda activate env
    ```

To check it worked, use:

=== "Linux, macOS, Windows Bash"

    ```console
    $ which pip

    some/directory/sysidentpy/env/Scripts/pip
    ```

=== "Windows PowerShell"

    ```console
    $ Get-Command pip

    some/directory/sysidentpy/env/Scripts/pip
    ```

=== "Windows Bash"

    ```console
    $ where pip

    some/directory/sysidentpy/env/Scripts/pip
    ```

If it shows the `pip` binary at `env/bin/pip` then it worked.



!!! tip
    Every time you install a new package with `pip` under that environment, activate the environment again.



!!! note
	We use the `pytest` package for testing. The test functions are located in tests subdirectories at each folder inside SysIdentPy, which check the validity of the algorithms.

#### Dependencies

Install SysIdentPy with the `dev` and `docs` option to get all the necessary dependencies to run the tests

=== "Dev and Docs dependencies"

    ``` sh
    pip install "sysidentpy[dev, docs]"
    ```

## Docs

First, make sure you set up your environment as described above, that will install all the requirements.

The documentation uses <a href="https://www.mkdocs.org/" class="external-link" target="_blank">MkDocs</a> and <a href="https://squidfunk.github.io/mkdocs-material/" class="external-link" target="_blank">Material for MKDocs</a>.

All the documentation is in Markdown format in the directory `./docs/`.

### Check the changes

During local development, you can serve the website locally and checks for any changes. This helps making sure that:

* All of your modifications were applied.
* The unmodified files are displaying as expected.


```console
$ mkdocs serve

INFO     -  [13:25:00] Browser connected: http://127.0.0.1:8000
```

It will serve the documentation on `http://127.0.0.1:8008`.

That way, you can keep editing the source files and see the changes live.

!!! warning
  If any modification break the build, you have to serve the website again. Always check your `console` to make sure you are serving the website.


## Run tests locally

Its always good to check if your implementations/modifications does not break any other part of the package. You can run the SysIdentPy tests locally using `pytest` in the respective folder to perform all the tests of the corresponding sub-packages.

#### Example of how to run the tests:

Open a terminal emulator of your choice and go to the main directory, e.g,

	\sysidentpy\

Just type `pytest` in the terminal emulator

```console
pytest
```

and you get a result like:

```console
========== test session starts ==========

platform linux -- Python 3.7.6, pytest-5.4.2, py-1.8.1, pluggy-0.13.1

rootdir: ~/sysidentpy

plugins: cov-2.8.1

collected 12 items

tests/test_regression.py ............ [100%]

========== 12 passed in 2.45s ==================
```
