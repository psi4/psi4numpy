## Hartree-Fock Self-Consistent Field

This module seeks to provide an overview of the theory and programming challenges associated with Hartree-Fock molecular orbital theory, with the following four tutorials:

2a) Restricted Hartree-Fock: Describes Hartree-Fock theory, and walks through a simple RHF program.
2b) Density Fitting: Introduces and describes the generation and manipulation of approximate 3-index ERI tensors, illustrated with a simple algorithm for building a Fock matrix.
2c) JK Building: Compares the speed and computational expense of several canonical and density-fitted algorithms for building Coulomb and Exchange matrices.
2d) Tensor Engines: Compares several engines for handling tensor contractions within quantum chemistry codes, including `np.einsum()`, `np.dot()`, and accessing direct BLAS routines through Psi4.
