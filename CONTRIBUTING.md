Thanks for considering contribute to Forte, a project of
the [ASYML family](https://asyml.io/).

This file outlines the guidelines for contributing to Forte and ASYML projects. While
the guideline cannot cover all scenarios, we ask everyone to be reasonable and make your
bets judgments, and feel free to propose changes to this document via
a [pull request](https://github.com/asyml/forte/pulls).

## Code of Conduct

This project and everyone participating in it is governed by
the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you
are expected to uphold this code. Please report unacceptable behavior
to [asyml.oss@gmail.com](mailto:asyml.oss@gmail.com.).

## How to contribute

### What can I contribute?

There are many ways you can contribute to ASYML projects. The goal of our projects is to
modularize Machine Learning and NLP questions, and to make NLP/ML problems as standard
engineering problems. Each project solves slightly different problems. Pick the problem
you are most interested and get started!

* [Texar](github.com/asyml/texar-pytorch)([Texar-Pytorch](github.com/asyml/texar)):
  Modularize a complex ML model into smaller components at different levels.
* [Forte](github.com/asyml/forte): Decompose and abstract complex NLP problems into
  multiple modules, and standardize the interface between the sub-problems and ML
  interface.
* [Forte Wrappers](github.com/asyml/forte-wrappers): Decompose and abstract complex NLP
  problems into multiple modules, and standardize the interface between the sub-problems
  and ML interface.
* [Stave](github.com/asyml/stave): Provide visualization and annotation for NLP tasks,
  by providing generic UI elements based on the abstraction.

### Forte and Forte-Wrapper Package Convention

We currently adopt a non-standard namespace packaging strategy. While this may introduce
some constraints in development, this allows us to provide a unified user experience.
The strategy, simply put, installs all Forte packages under "forte" module, but
currently stored in two different repositories, as different projects.

#### Core Packages in Forte Main

* forte: The root package contains the pipeline implementations, and defines
  the `pipeline_component`.
* forte.data: contains main data relevant components, mainly implements the data pack
  system and the ontology system.
* forte.dataset: contains readers to popular datasets.
* forte.processors: Processors are core components that perform NLP tasks. This module
  currently contains some base processor implementations and simple processor examples.
  We have
* forte.models: Contains our in-house development of some NLP models.
* forte.common: configuration, exceptions and other sharable code.

#### The `fortex` namespace and Forte Wrappers

* `fortex.xxx`: Forte Wrapper contains adapters of third party tools. Each tool is installed
  in its own namespace to avoid dependency conflicts. Each directory contains a standalone
  project and can be installed independently. *The project will be installed as
  `fortex.xxx` and under `fortex/xxx` folder in the site-packages.* For
  example, `fortex.nltk` will be installed under `site_packages/fortex/nltk` folder via
  `pip isntall forte.nltk` and the tool can be imported via `import fortex.nltk` and uninstalled
  via `pip uninstall forte.nltk`.

### Ontology namespaces
* The `ft.onto` namespace contains the core/basic ontology types defined by Forte, data types
  in this namespace are mainly generic NLP concepts, such as "Sentence", "Token".
* The `ftx` namespace supports namespace packaging:
  * We use `ftx.onto` namespace to show extra types for demo/example purposes.
  * We are also working one additional types in the `ftx.xxx` namespace types for certain domains.

### Report Bugs

Bugs are tracked as GitHub issues. Search
the [issues](https://github.com/asyml/forte-wrappers/issues) to make sure the problem is
not reported before.

To report a bug, create an issue provide the following information by filling in the bug
report template.

In the bug template, make sure you include enough information for reproducing the
problem:

* Use a descriptive title to identify the problem.
* Describe the steps to reproduce the problem, ideally a minimum code/command that can
  reproduce the problem.
* Describe the environment as detail as possible.
* Describe the actual behavior, and the expected behavior.

### Feature Request

Enhancements are also tracked as issues. Similarly, Search
the [issues](https://github.com/asyml/forte/issues) to make sure the enhancement is not
suggested before. To suggest the enhancement, create an issue by filling in the feature
enhancement template.

Following the feature template, fill in the information in more details:

* A clear and concise description of what the problem is.
* Describe the solution you'd like, with a clear and concise description of what you
  want to happen.
* Describe alternatives you've considered.
* Include as much context as possible.

### Pull Requests

When you have fixed a bug or implemented a new feature, you can create a pull request
for review.

* Use a [PR Template](https://github.com/asyml/forte/blob/master/.github/PULL_REQUEST_TEMPLATE.md) to structure your PR, and here:

  * The first line should always start with `This PR fixes [issue link]` or `This PR partially addresses [issue link]` where `[issue link]` can be replaced with a `#ISSUE_ID` associated to a specific [issue](https://github.com/asyml/forte/issues). This allows Github to automatically link your pull request to the corresponding issue. If this pull request will close the issue we use `fixes` otherwise we can say `partially addresses`.
  * **Description of changes**: You should include a description of the changes that you make. If the pull request is aimed to fix an issue, you can explain the approaches to address the problem.
  * **Possible influences of this PR**: List all the potential side-effects of your update. Examples include influences on compatibility, performance, API signature, etc.
  * **Test Conducted**: Describe the test cases to verify the changes in pull request. You should always create unit tests for your updates in the pull request and make sure they can cover all of the conditional branches, especially the ones related to corner cases where bugs usually stem from. We will use [Covergage](https://coverage.readthedocs.io/en/6.3/) to gauge the effectiveness of tests and you can refer to [Codecov report](https://about.codecov.io/language/python/) to see which lines are not visited by your test cases.
* Start your PR as draft and try to pass the Github Action CI check.
* Mark the PR to be ready-for-review once you are satisfied with it.

### Using Labels

We use standard issue labels such as priority, bug, enhancement, etc. We have a few
topic labels to identify the type of the issue. Currently the topics are `data` (
problems in our data system), `docs` (problems about documentation), `examples` (
problems about the examples), `interface` (the interfaces between different modules)
, `model` (machine learning models), `ontology` (the ontology system). We may have more
topic labels in the future.

## Style Guide

### Coding Style

The programming language for Forte is Python. We follow
the [Google Python Style guide](http://google.github.io/styleguide/pyguide.html). The
project code is examined using `pylint`, `flake8`, `mypy`, `black` and `sphinx-build` which will be run
automatically in CI. It's recommended that you should run these tests locally before submitting your pull request to save time. Refer to the github workflow [here](https://github.com/asyml/forte/blob/master/.github/workflows/main.yml) for detailed steps to carry out the tests. Basically what you need to do is to install the requirements (check out the `Install dependencies` sections) and run the commands (refer to the steps in `Format check with Black`, `Lint with flake8`, `Lint with pylint`, `Lint main code with mypy when torch version is not 1.5.0`, `Build Docs`, etc.).

We also recommend using tools `pre-commit` that automates the checking process before each commit since checking format is a repetitive process. We have the configuration file `.pre-commit-config.yaml` that lists several plugins including `black` to check format in the project root folder. Developers only need to install the package by `pip install pre-commit`. All the package versions in the `.pre-commit-config.yaml` must be consistent with package versions in [workflow configuration](https://github.com/asyml/forte/blob/master/.github/workflows/main.yml). For example, `black` package version should be set to the same.

### Docstring

 All public methods require docstring and type annotation. It is recommended to add docstring for all functions. The docstrings should follow the [`Comments and Docstrings` section](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) of Google Python Style guide. We will include a pylint plugin called [docparams](https://github.com/PyCQA/pylint/blob/main/pylint/extensions/docparams.rst) to validate the parameters of docstrings:
* parameters ~~and their types~~
  * types are only required in function signatures and sphinx will build parameter type hyperlinks based on that.
* return value and its type
* exceptions raised






You should take special care of the indentations in your documentation. Make sure the indents are consistent and follow the Google Style guide. All sections other than the heading should maintain a hanging indent of two or four spaces. Refer to the examples [here](https://google.github.io/styleguide/pyguide.html#383-functions-and-methods) for what is expected and what are the requirements for different sections like `args`, `lists`, `returns`, etc. Invalid indentations might trigger errors in `sphinx-build` and will cause confusing rendering of the documentation. You can run `sphinx-build` locally to see whether the generated docs look reasonable.

Another aspect that should be noted is the format of links or cross-references of python objects. Make sure to follow the [sphinx cross-referencing syntax](https://www.sphinx-doc.org/en/master/usage/restructuredtext/roles.html#xref-syntax). ~~The references will be checked by [sphinx-build nit-picky mode](https://www.sphinx-doc.org/en/master/man/sphinx-build.html#cmdoption-sphinx-build-n) which raises warnings for all the missing and unresolvable links.~~

### Jupyter Notebook
Notebooks are written under `docs/notebook_tutorial` folder, and we keep notebooks there for several reasons. First, it's friendly for new users to learn forte with a runnable notebook. Second, it can be rendered directly by the sphinx documentation pacakge by including the relative path to notebook in `docs/index_toc.rst`. It is straightforward for users to make references on how to use forte with the context of application. Third, we write notebook testing under `tests/forte/notebook` to ensure the notebook is runnable as API changes.

#### Notebook Rendering
Jupyter notebook written under `docs/notebook_tutorial` will be rendered in the sphinx documentation using package `nbsphinx`. You need to make sure notebook can be rendered normally in sphinx documentation. After writing notebook under , run this [command](https://github.com/asyml/forte/blob/ae3d46884c26bac95893cbbecfaf86168a039bdc/.github/workflows/main.yml#L135) under docs folder. It might give you some sphinx warnings and you need to fix them.

#### Notebook Hyperlinks
As notebook is rendered in the sphinx documentation, we might want to include hyperlinks to other sphinx pages in the documentation. For example, if we want to mention another `rst` file, we can write the hyperlinks in the markdown way with the relative path to the `rst` file such as `[reader](../toc/reader.rst)`.

#### Notebook Testing
As notebook includes code that might break over time when API changes. Plus, we want to test code efficiently.
Therefore, we test notebook by using package [`testbook`](https://testbook.readthedocs.io/en/latest/#).
User can refer to [notebook_test_tutorial.py](tests/forte/notebooks/notebook_test_tutorial.py) for how to test notebook.

##### Notebook Dependency
As notebook will be running in github CI, we need to consider its package dependencies and add required packages in `matrix.notebook-details.dep`.
As current notebooks requires fortex packages, so we limit torch version == 1.5.0 while testing notebooks.


#### Notebook Output
Notebooks will not be running automatically after committing files to the repository.
Developer needs to keep notebook cell outputs that are needed for the purpose of illustration.


### Git Commit Style

* Limit the first line to 72 characters or less
* Reference issues and pull requests in the second line
* For documentation only changes, use [skip ci] or [ci skip] in your commit messages to
  skip travis build.
