Self-consistent Field (SCF)
====================================

The following codes are available:
- `RHF.dat`: A simple Restricted Hartree-Fock (RHF) program.
- `RHF_DIIS.dat`: A RHF program that uses direct inversion of the iterative subspace (DIIS) to accelerate convergence.
- `RHF_libJK.dat`: A RHF program that uses Psi4's libJK to evaluate J and K matrices.
- `UHF_libJK.dat`: A UHF program that uses Psi4's libJK to evaluate J and K matrices.
- `CPHF.dat`: A Coupled-Perturbed RHF code for dipole polarizabilities.

Second-order SCF and Hessians:
- `SORHF.dat`: A second-order RHF program. Uses the electronic hessian to facilitate quadratic convergence.
- `SORHF.dat`: A second-order iterative RHF program.
- `SOROHF.dat`: A second-order ROHF program. Uses the electronic hessian to facilitate quadratic convergence.
- `SOROHF.dat`: A second-order iterative ROHF program.
- `SOUHF.dat`: A second-order UHF program. Uses the electronic hessian to facilitate quadratic convergence.
- `SOUHF.dat`: A second-order iterative UHF program.

Helper programs:
- `helper_HF.py`: A collection of helper classes and functions for Hartree-Fock.


Helper HF initialization:
```python
hf = helper_HF(psi4, energy, mol, memory=2, ndocc=None, scf_type='DF', guess='core'):
```
Input parameters:
- `psi4` - Global object.
- `energy` - Global object.
- `mol` - A Psi4 molecule object.
- `ndocc` - Number of occupied orbitals. If `None` the number of occupied orbitals is guessed based on nuclear charge.
- `scf_type` - ERI algorithm utilized to build J and K matrices.
- `guess` - Initial orbital selection.

Helper HF methods:
- `set_Cleft` - Sets alpha orbital, automatically builds density matrix.
- `diag` - Diagonalize matrix using `S^(-1/2)` orthogonalizer.
- `build_fock` - Builds the Fock matrix using current orbitals.
- `compute_hf_energy` - Computes the HF energy using current orbitals.
- `diis_add` - Adds a matrix to the DIIS vector, uses FDS - SDF for error vectors.
- `diis_update` - Updates orbitals using the DIIS method.
