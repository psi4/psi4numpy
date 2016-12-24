## Interactive Tutorials

These tutorials use the Jupyter notebook environment to offer an interactive, step-by-step introduction to several important concepts in quantum chemistry.  Their goal is to provide the reader with the necessary background to effectively program quantum chemical methods using the machinery provided by Psi4 and NumPy/SciPy.  

Below is a list of the available interactive tutorials, grouped by module:

1. Psi4 Basics
    * Molecule: Overview of the Molecule Class and coordinate input in Psi4
    * BasisSet: Building and manipulating basis sets within Psi4
    * Wavefunction: Introduction to wavefunction passing in Psi4

2. Hartree-Fock: Theory & Implementation
    * Restricted Hartree-Fock: Basic implementation of a self-contained RHF program
    * Density Fitting: Building approximate 2-electron integrals and an example density-fitted Fock matrix with Psi4 and the DFTensor object
    * JK Builds: Comparison of several algorithms for constructing Coulomb & Exchange matrices using full and density-fitted ERIs
    * Tensor Engines: Comparing relative algorithm speed when using `np.einsum()`, `np.dot()`, and direct BLAS calls with Psi4 for tensor contractions

3. Many Body Perturbation Theory 
    * MP2: Direct algorithm using full ERIs
    * DF-MP2: Density fitted algorithm

Note: these tutorials are under active construction.

Jupyter notebooks have the file extension ```.ipynb```.  In order to use these tutorials, Jupyter must first be installed.  Jupyter is available with the [Anaconda](https://www.continuum.io/downloads) python distribution.  Once installed, a Jupyter notebook ```example.ipynb``` may be opened from the command line with
```
jupyter-notebook example.ipynb
```
