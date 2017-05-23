Contributing to Psi4NumPy
=========================

Welcome to the Psi4NumPy Developer Community! 

The single largest factor which makes this repository an effective resource is
that it was inspired by, designed for, and maintained by the quantum &
computational chemistry community.  Therefore, your contributions to this
project are vital -- through reporting issues, contributing to discussions on
posts and pull requests, and/or submitting code of your own, you can impact how
this project moves forward and how well we serve yours as well as the needs of
the community. 

The following are a set of suggestions for the best way to contribute to the
Psi4NumPy project. Just like the Code of the Pirate Brethren set forth by Morgan and
Bartholomew, they're more what you'd call "guidelines" than actual rules.
Hopefully, however, this document (and the guidelines inside) will make
contributing to Psi4NumPy both easier and more effective for everyone.  

Good luck & happy coding!

-- The Psi4NumPy Developers 

#### Table of Contents

1. [What do I need to know before starting to contribute to Psi4NumPy?](#what-do-i-need-to-know-before-starting-to-contribute-to-psi4numpy)
    1. [Project Goals](#project-goals)
    2. [Repository Organization](#repository-organization)
2. [How can I contribute to Psi4NumPy?](#how-can-i-contribute-to-psi4numpy)
    * [Reference Implementations](#reference-implementations)
    * [Interactive Tutorials](#interactive-tutorials)
    * [Value Comparison](#value-comparison) 
3. [Styleguides](#styleguides)
    1. [Python Styleguide](#python-styleguide)
    2. [Git Commit Messages](#git-commit-messages)
    3. [Citation Styleguide](#citation-styleguide)
    4. [Header Styleguide](#header-styleguide)
    5. [Documentation Sytleguide](#documentation-styleguide)
4. [Possible development workflow](#possible-development-workflow)

# What do I need to know before starting to contribute to Psi4NumPy?

## Project Goals

The strategic goals of the Psi4NumPy project are to provide the quantum chemistry community with
* a programming environment for rapid prototyping and mehod development,
* a repository for reference implementations of existing and novel quantum chemical methods, and
* an interactive educational framework which combines theory and implementation to effectively teach the
programming of quantum chemical methods.

For more information on these goals, refer to the [reference
implementation](#reference-implementations) and [interactive
tutorial](#interactive-tutorials) sections below, as well as the project
overview [here](https://github.com/psi4/psi4numpy#overview).

## Repository Organization

This repository contains

* reference implementations, which provide working Python scripts
implementing various quantum chemical methods, and
* interactive tutorials, which provide Jupyter notebooks presenting a hybrid 
theory-and-implementation educational framework for learning to program quantum 
chemistry methods.  

Reference implementations are organized into top-level directories
corresponding to the over-arching theory upon which each method is based, i.e.,
both EOM-CCSD and TD-CCSD are contained in the
[Coupled-Cluster](https://github.com/psi4/psi4numpy/tree/master/Coupled-Cluster)
directory.  All interactive tutorials are contained in the top-level directory
[Tutorials](https://github.com/psi4/psi4numpy/tree/master/Tutorials).  These
tutorials are organized in logical order of progression, which is enumerated in
detail
[here](https://github.com/psi4/psi4numpy/tree/master/Tutorials#interactive-tutorials).

# How can I contribute to Psi4NumPy?

There are several ways to become involved with Psi4NumPy, including 

* participating in discussions on [pull
requests](https://github.com/psi4/psi4numpy/pulls) and
[issues](https://github.com/psi4/psi4numpy/issues),
* updating/adding features to existing content, and
* submitting new content to the repository.

Each of these possible contributions are highly valuable, and will help Psi4NumPy to
better serve the needs of the quantum/computational chemistry community at
large. Therefore, your contribution is crucial to the success of this project!

Below, guidelines for submitting new content are discussed, first for
[reference implementations](#reference-implementations) and then for
[interactive tutorials(#interactive-tutorials). For convenience, a possible
development workflow is given [here](#suggested-workflow).

## Reference Implementations

Reference implementations of quantum chemical methods should take the form of a
self-contained Python script which presents a complete program that
successfully implements the target method. Each such script should contain 

* an appropriate [header](#header-styleguide), 
* code with appropriate [Python style convention](#python-styleguide), 
* [references](#citation-styleguide) to relevant resources (algorithms, publications, etc.), and 
* [value comparison](#value-comparison) to ensure proper code function.  

In some cases, particularly with SAPT, CC, and CI theories, a fully
self-contained script may be long and complex enough that a self-contained
script may be too extensive to retain maximum readability, or multiple scripts
within the same method directory might benefit from some central machinery.  To
address both of these points:

1. for post-Hartree--Fock methods, call Psi4 for the HF energy & wavefunction with
```
scf_e, scf_wfn = psi4.energy('scf', return_wfn=True)
```
before implementing the target method, or 
2. creating a (or adding to an existing) method helper module which contains
helper classes and functions that refactors some of the method's
implementation.

If attempting to refactor the implementation to make use of a helper
module/class, it is critical to ensure that the reference implementation itself
is still explicit -- after all, one of the goals of Psi4NumPy is to provide
clear, readable implementations, even at the expense of program speed.

## Interactive Tutorials

Interactive tutorials provide a hybrid theory-and-implementation educational
framework within a Jupyter notebook for the reader to learn general information
about a method's formulation and detailed information with respect to the
method's implementation.  A good tutorial is not just about clear, correct code
(like the reference implementations) but also requires that each detail be
motivated logically and explained fully. Each tutorial contains 

* an appropriate [header](#header-styleguide), 
* a [theory overview](#theory-overview), 
* an [implementation](#implementation), 
* a testable [value comparison](#value comparison), and 
* a list of relevant [citations](#citation-styleguide).  

#### Theory Overview

As the purpose of these tutorials is not to completely derive a method, it is
better to provide a broad-strokes overview of the formulation of the method, as
well as explicitly providing (or referencing) all equations to be implemented.
Fortunately, markdown cells within Jupyter notebooks support LaTeX math formatting,
and GitHub is capable of rendering these equations when viewing the notebook.

#### Implementation

This section should describe the details of the implementation, with particular
focus on *why* a certain algorithm or coding decision is made, not just what
the decision is.  For example, the naive O(N^8) integral transformation within
MP2 is discussed in the conventional MP2 tutorial
[here](https://github.com/psi4/psi4numpy/blob/master/Tutorials/05_Moller-Plesset/5a_conventional-mp2.ipynb)
before the factored O(N^5) transformation is introduced.  This section benefits
from alternating markdown and code cells, so that the reader can execute the
cells alongside reading about the implementation.  Again, specific equations
within publications should be referenced with [LNFA:yy:pp] keys, which should
match those references within the `Tutorial References` section (see
[below](#citation-styleguide)).

Note: Every tutorial should be accompanied by a corresponding reference
implementation.  This is because the interactive tutorials are like a
classroom, while the reference implementation is like a textbook.  While a more
adventurous student may skip class to rely on reading the book, every class
needs an accompanying textbook for supporting material.  The accompanying
reference implementation can be as simple as reproducing the code cells in a
single script, or potentially more complex if supplemented by helper
functions/classes/modules.

### Value Comparison

To ensure the continued health of contributed reference implementations and
interactive tutorials as Psi4 and NumPy development continues, provide one or
more `psi4.compare_values()` statements (documented
[here](http://psicode.org/psi4manual/1.1/api/psi4.driver.compare_values.html?highlight=compare_values#psi4.driver.compare_values))
toward the end of your scripts/tutorials. These can contain a hard-coded
reference value or (if available) an equivalent pure-Psi4 computation:

```python
psi4.compare_values(expected, computed, digits, label)
```

where `expected` is the reference value, and `computed` is the final output of the
script/tutorial.  

## Styleguides

### Python Styleguide

All Python code should should conform to the
[PEP8](https://www.python.org/dev/peps/pep-0008/) Python style guide
conventions.  This code should be compliant with Python versions 2.7/3.5/3.6,
meaning that `print` must be called as a function, i.e., `print('Compliant with
Python 3')` and not `print 'This will break Python 3!'`.

### Git Commit Messages

In addition to the conventions for writing good git commit messages
[here](https://chris.beams.io/posts/git-commit/), commits to Psi4NumPy should
add the following codes to the beginning of the subject line to categorize the
nature of the commit:

| Code | Category Description |
-------|-----------------------
| RIMP | New reference implementation |
| TUT | New interactive tutorial |
| DEV | New developer tool or utility |
| ENH | Enhancement of any existing implementations or tutorials |
| UPD | Update to new Psi4/NumPy/SciPy compatibility |
| MAINT | Maintenance commit (typos, repo reorganization, etc.) |
| STY | Style convention changes (PEP8, headers, etc.) |
| TST | Related to tests of the codebase |
| REF | Adding to/modifying any references/citations in tutorials, scripts, or READMEs |
| DOC | Documentation (README additions, etc. other than REF) |
| REL | Related to the release of Psi4NumPy |

### Citation Styleguide

For both reference implementations and interactive tutorials, a complete list
of citations from which the content was drawn must be provided.  Such citation
lists should contain entries of the form (in Markdown):
```
1. [LNFA:yy:pp](https://link-to-publication-website.com) J. Doe *et al.* *J. Abbrev.* **Issue**, pages (year)
```
The LNFA:yy:pp citation key contains the last name of the first author (LNFA),
the publication year (yy), and the publication pages (pp), and the full
citation should be supplied in [AIP
format](https://publishing.aip.org/authors/preparing-your-manuscript) (see link
section Refernces -- By Number).  

For reference implementations, citations should appear 

* as full list items in the "References" section of the relevant method
directory's `README.md` file (see [Repository
Organization](#repository-organization) above), and 
* as keys followed by relevant equation information in code comments/function docstrings.  

This final point is important when specific equations from a publication are
implemented within the script.  In this case, a Python comment containing this
publication's citation key and the equation number for the expression
beingimplemented should be given nearby.  For example, to implement a function
which builds the `Wmnij` intermediate from equation 6 of [J. F.  Stanton *et
al.*'s paper](http://dx.doi.org/10.1063/1.460620), 
```python
def build_Wmnij():
    """Implements [Stanton:1991:4334] Eqn. 6"""
    code code code
```

For interactive tutorials, unlike for reference implementations, a references
section within each tutorial should be included in a markdown cell at the very
end of the notebook.  This cell should contain a citation list of the form
above.  Since the tutorials are effectively standalone entities, however, this
`Tutorial References` section should *not* be augmented with a central
references section within the method's README.  Additionally, specific
equations should be referenced within tutorials using the citation key for the
appropriate publication.

TL;DR: Citations, including equation numbers, should be provided in close
proximity (docstrings or comments) to the code. It can also be handy to collect
all citations into the nearest README.md (for reference implementations) or
final cell (for tutorials).

### Header Styleguide

Each reference implementation script and tutorial notebook should begin with a
statement about the purpose of the script, author and contributor information,
and copyright, license, and date information:

```python
"""A simple explanation of script contents."""

__authors__   =  "John and Jane Doe"
__credits__   =  ["John Doe", "Jane Doe", "John Q. Sample"]
__email__     =  "jdoe@example.com"

__copyright__ = "(c) 2014-2017, The Psi4NumPy Developers"
__license__   = "BSD-3-Clause"
__date__      = "5/22/2017"
```

The difference between `__authors__` and `__credits__` is largely semantic;
one possible delineation can be summarized by

* `__author__`: individuals who have written/modified large parts of the script/code
* `__credits__` : individuals who cleaned up or modified the code slightly.

In this way, all contributions are recognized and retained, no matter the scope.

### Documentation Styleguide

Documentation for the contents of the Psi4NumPy repository is contained in
`README.md` files in each directory.  

For reference implementations, `README.md` files in the `method-dir` should contain
* a *very* brief overview of the most important aspects of the theory,
* a description of the primary arrays, variables, and indices for the method,
* a description of the proper use and capabilities of any helper classes or modules, and
* a list containing relevant citations for all scripts in the `method-dir` (see [above](#citation-styleguide))
See the SAPT README [here](https://github.com/psi4/psi4numpy/blob/master/Symmetry-Adapted-Perturbation-Theory/README.md)
for an example.

For interactive tutorials, two levels of `README.md` files exist.  First, the
`README` for the main `Tutorials` folder
[here](https://github.com/psi4/psi4numpy/tree/master/Tutorials) contains a list
of available tutorials grouped by module.  Second, in each
`module-dir/README.md`, a more detailed list of the module and individual
tutorial contents is given.  See the Hartree--Fock module README
[here](https://github.com/psi4/psi4numpy/blob/master/Tutorials/03_Hartree-Fock/README.md)
for an example.

## Possible Development Workflow

This section outlines a possible workflow for contributing to Psi4NumPy,
comprised of the following stages:

1. [Creating your own fork of the Psi4NumPy repository](#creating-your-own-fork-of-the-psi4numpy-repository),
2. [Creating a new feature branch](#creating-a-new-feature-branch) for your content,
3. [Creating and committing your content](#creating-and-committing-your-content),
4. [Opening a pull request](#opening-a-pull-request) to merge your content into the parent repository,
5. [Testing and continuing integration](#testing-and-continuous-integration) of your content,
6. [Code review](#code-review) with the Psi4NumPy developer community, and
7. [Merging your content](#merging-your-content) into the main Psi4NumPy repository.

For more information, the suggested workflow for contributing to the Psi4
repository
[here](http://psicode.org/psi4manual/1.1/build_obtaining.html#what-is-the-suggested-github-workflow)
and the GitHub guide [here](https://guides.github.com/introduction/flow/)
provide graphical representation which visualizes this process.

Note: For clarity when demonstrating Git commands, we will indicate the
(current-branch) and relative filepath to the `top-level-psi4numpy-dir` within
the shell prompt.

#### Creating your own fork of the Psi4NumPy repository

To create a fork of the Psi4NumPy repository within your GitHub account, follow
along with the steps below.  This guide was adapted for our repo from the
GitHub user documentation
[here](https://help.github.com/articles/fork-a-repo/).

1. On the main [Psi4NumPy GitHub](https://github.com/psi4/psi4numpy) page, hit the [Fork](https://help.github.com/articles/fork-a-repo/) button to create a fork of the repository on your GitHub account.
2. From a terminal on your local computer (replacing `johndoe` with your GitHub username):
```
# Clone fork `johndoe/psi4numpy` to local
~$ git clone https://github.com/johndoe/psi4numpy.git

# Navigate into your top-level-psi4numpy-dir
~$ cd psi4numpy

# Add a connection between your forked repository and the parent repository
~psi4numpy (master)$ git remote add upstream https://github.com/psi4/psi4numpy.git

# Verify origin & upstream
~psi4numpy (master)$ git remote -v
origin    https://github.com/johndoe/psi4numpy.git (fetch)
origin    https://github.com/johndoe/psi4numpy.git (push)
upstream    https://github.com/psi4/psi4numpy.git (fetch)
upstream    https://github.com/psi4/psi4numpy.git (push)
```

If you have already cloned locally from the `psi4/psi4numpy` master branch:
1. Create a new fork on your GitHub account according to (1) above
2. Verify location of `origin`:
```
~psi4numpy (master)$ git remote -v
origin    https://github.com/psi4/psi4numpy.git (fetch)
origin    https://github.com/psi4/psi4numpy.git (push)
```
3. If the two lines printed by `git remote -v` match the above:
```
# Reset origin to johndoe/psi4numpy:origin
~psi4numpy (master)$ git remote set-url origin https://github.com/johndoe/psi4numpy.git

# Verify origin reset
~psi4numpy (master)$ git remote -v
origin    https://github.com/johndoe/psi4numpy.git (fetch)
origin    https://github.com/johndoe/psi4numpy.git (push)

# Connect your fork with parent repository
~psi4numpy (master)$ git remote add upstream https://github.com/psi4/psi4numpy.git

# Verify origin & upstream
~psi4numpy (master)$ git remote -v
origin    https://github.com/johndoe/psi4numpy.git (fetch)
origin    https://github.com/johndoe/psi4numpy.git (push)
upstream    https://github.com/psi4/psi4numpy.git (fetch)
upstream    https://github.com/psi4/psi4numpy.git (push)
```

#### Creating a New Feature Branch

Branches are the defining feature of the Git verisioning software.  While
branches can become complicated,
[documentation](psi4nump://git-scm.com/book/en/v1/Git-Branching) and
[tutorials](http://learngitbranching.js.org/) exist elsewhere.  The most
critical idea is that the `master` branch (the repository's default branch) is
always deployable.  Great lengths must therefore be taken to protect the
integrity of the `master` branch, the first of which is to conduct content
development on a different branch than `master`.  To create a new branch on which
to develop the content "new-feature",
```
# Create branch new-feature and change to it
~psi4numpy (master)$ git checkout -b new-feature
~psi4numpy (new-feature)$
```

#### Creating and committing your content

Guidelines for creating [reference implementations](#reference-implementations)
and [interactive tutorials](#interactive-tutorials), as well as styleguides for
writing [documentation](#documentation-styleguide) and [citation
lists](#citation-styleguide), are given below.  After your content has been
created, it must be added and committed to your branch and pushed to your fork.
To add new and/or modified files to the stage:
```
# Add modified and new files to stage
~psi4numpy (new-feature)$ git add file1 file2
```
Next, to commit your changes, use `git commit`.  This will open the default text editor,
in which you will write a brief [commit message](#git-commit-messages) describing your content:
```
# Commit changes, write commit message in text editor
~psi4numpy (new-feature)$ git commit
[new-feature 4ade581] DEV: Added a new feature
 2 files changed, 30 insertions(+), 674 deletions(-)
```
Finally, push your content to your fork
```
# Push to your fork, creating remote branch origin:new-feature
~ (new-feature)$ git push origin new-feature
```

#### Opening a Pull Request

Follow the instructions
[here](https://help.github.com/articles/creating-a-pull-request/) and
[here](https://help.github.com/articles/creating-a-pull-request-from-a-fork/)
for creating a GitHub pull request from the `new-feature` branch of your fork
`johndoe/psi4numpy` to the `master` branch of the `psi4/psi4numpy` repository
(`upstream` above).  A template pull request message will be provided for you,
where you can describe the new feature you are requesting be added to Psi4NumPy.

#### Testing and Continuous Integration

All pull request content will be tested with continuous integration to ensure that

* submitted code runs successfully, and
* provided value comparison tests pass successfully.

This process will begin automatically when a new pull request is opened, and merging will
be blocked if continuous integration testing fails.

#### Code Review

In addition to the testing by continuous integration, all contributed content
to the Psi4NumPy repository is subject to code review by the developer
community.  This allows for a collaborative discussion between developers and
the pull request author, which can improve the quality, clarity, and/or goals
of a particular PR on an individual basis.  In this way, the repository is
maintained by the entire developer community every step of the way.  During
this process, one or several core developers (devs with write access to the
repository) will review the pull request and either accept it or request
changes which must be addressed by the author before merging is allowed.  

#### Merging your Content

Once a pull request passes continuous integration testing and a reviewer has
approved the pull request, it will be merged into the Psi4NumPy branch
`master`.  Upon merging, your branch `new-feature` will become obscolete; it
can be deleted either with an available button on the GitHub website or by
executing the following at the command line:

```
# Switch back to master branch
~psi4numpy (new-feature)$ git checkout master

# Delete new-feature branch on local
~psi4numpy (master)$ git branch -d new-feature

# Delete origin:new-feature branch on remote
~psi4numpy (master)$ git push origin :new-feature
```

Finally, update your fork branch `master` to include your newly incorporated
content:

```
# Pull from upstream master
~psi4numpy (master)$ git pull upstream master

# Push to your fork branch `master` (may need to add --force)
~psi4numpy (master)$ git push origin master
```

