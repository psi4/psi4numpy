# Response theory

## Self-consistent field (SCF)

- `CPHF.py`: A Coupled-Perturbed RHF code for dipole polarizabilities.
- `beta.py`: A Coupled-Perturbed RHF code for first dipole hyperpolarizabilities using the $2n+1$ rule.
- `TDHF.py`: An RHF code for solving the TDHF/RPA equations for linear response excitation energies and C6 coefficients.

Helper programs:
- `helper_CPHF.py`: A collection of helper classes and functions for Coupled-Perturbed Hartree-Fock. Includes both frequency-dependent direct electronic Hessian inversion and iterative solvers.

## Coupled-Cluster Linear Response (CCLR)

Inside RHF/CCSD:
- `polar.py`: A simple script for calculating CCSD dipole polarizabilities using Coupled Cluster Linear Response (CCLR) theory.
- `optrot.py`: Computing specific optical rotation using Coupled Cluster Linear Response
- `helper_ccpert.py`: Helper classes and functions for CCLR implementations.

