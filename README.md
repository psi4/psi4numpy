<p align="center">
<br>
<img src="media/psi4banner_numpy_interactive.png" alt="Psi4NumPy banner logo" height=200> <br>
<a href="https://zenodo.org/badge/latestdoi/22622192"> <img src="https://zenodo.org/badge/22622192.svg" /></a>
<a href="https://travis-ci.org/psi4/psi4numpy"><img src="https://travis-ci.org/psi4/psi4numpy.svg?branch=master"></a>
<br>
</p>

---

### Overview

The overall goal of the Psi4NumPy project is to provide an interactive quantum chemistry
framework for reference implementations, rapid prototyping, development, and education.
To do this, quantities relevant to quantum chemistry are computed with the
<span style="font-variant:small-caps;"> Psi4 </span> electronic structure package, and subsequently manipulated 
using the Numerical Python (NumPy) package.  This combination
provides an interface that is both simple to use and remains relatively fast
to execute. 

A series of short scripts demonstrating the implementation of Hartree-Fock Self-Consistent 
Field, SCF Response, Møller-Plesset Perturbation Theory, Symmetry-Adapted Perturbation Theory, 
Coupled Cluster Theory, and more are provided for the reference of the quantum chemistry
community at large to facilitate both reproducibility and low-level methodological understanding.
Additionally, the Tutorials folder above represents an interactive educational
environment containing modules discussing the theory and implementation of various
quantum and computational chemistry methods.  By leveraging the popular Jupyter Notebook
application, each tutorial is constructed as hybrid theory and programming in an easy to use
interactive environment, removing the gap between theory and implementation.

If you have comments, questions, or would like to contribute to the project
please see our [contributor guidelines]().

### Getting Started

1. Obtain required software:
- [Psi4](http://psicode.org/psi4manual/1.1/build_obtaining.html)
- [Python](https://python.org) 2.7+
- [NumPy](http://www.numpy.org) 1.7.2+
- [Scipy](https://scipy.org) 0.13.0+
 
2. Update Psi4 to the development version with Conda:
```
conda update psi4 -c psi4/label/dev
```

3. Link PsiAPI
    1. Find appropriate paths
```
~$ bash
~$ psi4 --psiapi-path
export PATH=$HOME/psi4conda/bin:$PATH
export PYTHONPATH=$HOME/psi4conda/lib/python-x.x/site-packages:$PYTHONPATH
```
    2. Export relevant paths
```
~$ export PATH=$HOME/psi4conda/bin:$PATH
~$ export PYTHONPATH=$HOME/psi4conda/lib/python-x.x/site-packages:$PYTHONPATH
```
4. Run scripts as conventional Python scripts, ```python -c "import psi4"```.

New users can follow the
[Getting Started](https://github.com/psi4/psi4numpy/blob/master/Tutorials/01_Psi4NumPy-Basics/1a_Getting-Started.ipynb)
notebook or the [PsiAPI documentation](http://psicode.org/psi4manual/master/psiapi.html) for an introduction to running Psi4 within the PsiAPI.

A tutorial that covers the basics of NumPy can be found
[here](http://wiki.scipy.org/Tentative_NumPy_Tutorial).

### Repository Organization

This repository contains

* reference implementations, which provide working
programs implementing various quantum chemical methods as Python scripts, and
* interactive tutorials, which provide a hybrid theory-and-implementation
educational framework for learning to program quantum chemistry methods as Jupyter
notebooks.

Reference implementations are organized into top-level directories
corresponding to the over-arching theory upon which each method is based, i.e.,
both EOM-CCSD and TD-CCSD are contained in the
[Coupled-Cluster](https://github.com/psi4/psi4numpy/tree/master/Coupled-Cluster)
directory.  All interactive tutorials are contained in the top-level directory
[Tutorials](https://github.com/psi4/psi4numpy/tree/master/Tutorials).  These
tutorials are organized in logical order of progression, which is enumerated in
detail
[here](https://github.com/psi4/psi4numpy/tree/master/Tutorials#interactive-tutorials).

### Psi4 v1.1
This repostitory has recently been updated to be compatible with Psi4 version 1.1.
Please see the `v1.0` branch for a Psi4 v1.0 compliant Psi4NumPy version. 
