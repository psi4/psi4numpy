![Psi4NumPy](media/psi4banner_numpy.png)
=============

#####Author: Daniel G. A. Smith
#####Contact: dgasmith@gatech.edu

#####Psi4 v1.1
Note: This repostitory has drifted over time from its original intent. A
cleanup pass and update to Psi4 version v1.1 is taking place. Please see the
`v1.0` branch for a Psi4 v1.0 compliant Psi4NumPy version. 

#####Overview

The overall goal of Psi4NumPy is to provide clear, readable code for both
learning and development. Python is used to "plug" together low level languages
and provide an interface that is both simple to use and remains relatively fast
to execute. In this case Psi4 is used for molecular properties and integrals,
NumPy/scipy is used for tensor operations and linear algebra.

First generation tutorials can be found in the `Tutorials` folder and provide
the basic idea behind learning with the Psi4NumPy approach.  New interactive
tutorial are found in the `Interactive-Tutorials` folder. These tutorials are
what this project aims to be, hybrid theory and programming in an easy to use
interactive environment. Finally, the remaining folders contain an assortment
of scripts that describe how to program a variety of quantities.  These scripts
are unfortunately light on the associated theory involved; however, this should
change over time.

All scripts within this repository have been reworked to be generic Python
scripts.  To import Psi4 locate the `Psi4_install/lib` directory and add this
to your Python path: `export PYTHONPATH=Psi4_install/lib`. All scripts should
then be run as conventional Python scripts, `python -c "import Psi4"`.

If you have comments, questions, or would like to contribute to the project
please feel free to email [me](mailto:dgasmith@gatech.edu).

A tutorial that covers the basics of NumPy can be found
[here](http://wiki.scipy.org/Tentative_NumPy_Tutorial).

#####Requirements:
- [Psi4](https://github.com/Psi4/Psi4) 1.1+
- [Python](python.org) 2.7+
 - [NumPy](scipy.org) 1.7.2+
 - [Scipy](numpy.scipy.org) 0.13.0+

