Interactive Tutorials
=====================

These tutorials use the Jupyter notebook environment to offer an interactive,
step-by-step introduction to several important concepts in quantum chemistry.
Their goal is to provide the reader with the necessary background to
effectively program quantum chemical methods using the machinery provided by
Psi4 and NumPy/SciPy.  

Below is a list of the available interactive tutorials, grouped by module:

1. Psi4NumPy Basics
    * Molecule: Overview of the Molecule class and geometry specification in Psi4
    * Psi4-NumPy Data Sharing: Describes the integration of Psi4 Matrix and NumPy array objects
    * Wavefunction: Building, passing, and using wavefunction objects from Psi4
    * MintsHelper: Generating one- and two-electron integrals with Psi4
    * Tensor Manipulation: Overview of commonly used tensor engines throughout Psi4NumPy 
    * BasisSet: Building and manipulating basis sets within Psi4

2. Linear Algebra

3. Hartree-Fock Molecular Orbital Theory
    * Restricted Hartree-Fock: Implementation of the closed-shell, restricted orbital formulation of Hartree-Fock theory
    * Direct Inversion of the Iterative Subspace: Theory and integration of the DIIS convergence acceleration method into an RHF program
    * Unrestricted Hartree-Fock: Implementation of the open-shell, unrestricted orbital formulation of Hartree-Fock theory, utilizing DIIS convergence acceleration
    * Density Fitting: Overview of the theory and construction of density-fitted ERIs in Psi4, and an illustrative example of a density-fitted Fock build.

4. Density Functional Theory (requires Psi4 1.2, beta)
    * DFT Grid: Briefly outlines several details of the DFT Grid.
    * LDA Kernel: Discusses how the local density approximation (LDA) is formed and computed.
    * GGA and Meta Kernels: Focuses on higher-rung DFT variants and examines how to compute various functionals.
    * VV10: A guide how to compute the non-local kernel VV10.

5. Møller–Plesset Perturbation Theory 
    * Conventional MP2: Overview and implementation of the conventional formulation of second-order Møller-Plesset Perturbation Theory (MP2), using full 4-index ERIs.
    * Density Fitted MP2: Discusses the implementation of the density-fitted formulation of MP2 with an efficient, low-memory algorithm utilizing permutational ERI symmetry.

13. Geometry Optimization Theory
    * Internal coordinates: the B-matrix and coordinate transformations.
    * Hessians: estimating and transforming second derivatives.
    * Hessians: update schemes with the example of BFGS.
    * RFO: Rational Function Optimization minimizations.
    * The Step-Backtransformation: From displacements in internals to displacements in Cartesians.



Note: These tutorials are under active construction.

Jupyter notebooks have the file extension `.ipynb`.  In order to use these
tutorials, Jupyter must first be installed.  Jupyter is available with the
[Anaconda](https://www.continuum.io/downloads) python distribution.  Once
installed, a Jupyter notebook `example.ipynb` may be opened from the command
line with
```
jupyter-notebook example.ipynb
```

These modules and the tutorials contained therein make use of advanced
scientific Python programming with NumPy/SciPy, and assume familiarity with
these packages to focus more closely on the intricacies of programming quantum
chemistry.  Before jumping into Module 1, it is therefore advantageous to at
the very least skim through the [NumPy quickstart tutorial](https://docs.scipy.org/doc/numpy-dev/user/quickstart.html) and 
[SciPy tutorial](https://docs.scipy.org/doc/scipy/reference/tutorial/index.html).  For a
more thorough introduction to these two packages, please refer to the [SciPy
Lectures](http://www.scipy-lectures.org/).  Good luck and happy
programming!
