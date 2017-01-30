## Interactive Tutorials

These tutorials use the Jupyter notebook environment to offer an interactive, step-by-step introduction to several important concepts in quantum chemistry.  Their goal is to provide the reader with the necessary background to effectively program quantum chemical methods using the machinery provided by Psi4 and NumPy/SciPy.  

Below is a list of the available interactive tutorials, grouped by module:

1. Psi4NumPy Basics
    * Molecule: Overview of the Molecule class and coordinate input in Psi4
    * BasisSet: Building and manipulating basis sets within Psi4
    * Wavefunction: Introduction to the Wavefunction class in Psi4

2. Linear Algebra

3. Hartree-Fock: Theory & Implementation
    * Restricted Hartree-Fock: Basic implementation of a self-contained RHF program
    * Direct Inversion of the Iterative Subspace: Theory and integration of the DIIS convergence acceleration method into an RHF program
    * Unrestricted Hartree-Fock: Implementation of the open-shell, unrestricted orbital formulation of Hartree-Fock theory, utilizing DIIS convergence acceleration
    * Density Fitting: Building approximate 2-electron integrals and an example density-fitted Fock matrix with Psi4 and the DFTensor object
    * JK Builds: Comparison of several algorithms for constructing Coulomb & Exchange matrices using full and density-fitted ERIs
    * Tensor Engines: Comparing relative algorithm speed when using `np.einsum()`, `np.dot()`, and direct BLAS calls with Psi4 for tensor contractions

4. Density Functional Theory (requires Psi4 1.2, beta)

5. Møller–Plesset Perturbation Theory 
    * Conventional MP2: Overview of Moller-Plesset Perturbation Theory and implementation of a simple second-order Moller-Plesset (MP2) program using conventional, 4-index ERIs.
    * Density Fitted MP2


Note: These tutorials are under active construction.

Jupyter notebooks have the file extension `.ipynb`.  In order to use these tutorials, Jupyter must first be installed.  Jupyter is available with the [Anaconda](https://www.continuum.io/downloads) python distribution.  Once installed, a Jupyter notebook `example.ipynb` may be opened from the command line with
```
jupyter-notebook example.ipynb
```

These modules and the tutorials contained therein make use of advanced scientific Python programming with NumPy/SciPy, and assume familiarity with these packages to focus more closely on the intricacies of programming quantum chemistry.  Before jumping into Module 1, it is therefore advantageous to at the very least skim through the NumPy quickstart tutorial [here](https://docs.scipy.org/doc/numpy-dev/user/quickstart.html) and SciPy tutorial [here](https://docs.scipy.org/doc/scipy/reference/tutorial/index.html).  For a more thorough introduction to these two packages, please refer to the SciPy Lectures [here](http://www.scipy-lectures.org/).  Good luck and happy programming!
