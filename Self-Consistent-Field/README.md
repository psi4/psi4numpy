Self-consistent Field (SCF)
====================================

The following codes are available:
- `RHF.dat`: A simple RHF program.
- `RHF_DIIS.dat`: A RHF program that uses direct inversion of the iterative subspace (DIIS) to accelerate convergence.
- `RHF_libJK.dat`: A RHF program that uses Psi4's libJK to evaluate J and K matrices.
- `SORHF.dat`: A second-order RHF program. Uses the electronic hessian to facilitate quadratic convergence.
- `helper_HF.py`: A HF helper program.

Helper HF initialization:
```python
hf = helper_HF(psi4, energy, mol, memory=2, ndocc=None, scf_type='DF', guess='core'):
```
Input parameters:
- `psi4` - Global object.
- `energy` - Global object.
- `mol` - A Psi4 molecule object.
- `ndocc` - Number of occupied orbitals. If `None` the number of occupied orbitals is guess based of nuclear charge.
- `scf_type` - ERI algorithm utilized to build J and K matrices.
- `guess` - Initial orbital selection.

Helper HF methods:
- `set_Cleft` - Sets alpha orbital, automatically builds density matrix.
- `diag` - Diagonalize matrix using `S^(-1/2)` orthogonalizer.
- `build_fock` - Builds the Fock matrix using current orbitals.
- `compute_hf_energy` - Computes the HF energy using current orbitals.
- `diis_add` - Adds a matrix to the DIIS vector, uses FDS - SDF for error vectors.
- `diis_update` - Updates orbitals using the DIIS method.
