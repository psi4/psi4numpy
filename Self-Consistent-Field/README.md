Self-consistent Field (SCF)
====================================

The following codes are available:
- `RHF.py`: A simple Restricted Hartree-Fock (RHF) program.
- `RHF_DIIS.py`: A RHF program that uses direct inversion of the iterative subspace (DIIS) to accelerate convergence.
- `RHF_libJK.py`: A RHF program that uses Psi4's libJK to evaluate J and K matrices.
- `ROHF_libJK.py`: A ROHF program that uses Psi4's libJK to evaluate J and K matrices.
- `UHF_libJK.py`: A UHF program that uses Psi4's libJK to evaluate J and K matrices.

Second-order SCF and Hessians:
- `SORHF.py`: A second-order RHF program. Uses the electronic Hessian to facilitate quadratic convergence.
- `SORHF_iterative.py`: A second-order iterative RHF program.
- `SOROHF.py`: A second-order ROHF program. Uses the electronic Hessian to facilitate quadratic convergence.
- `SOROHF_iterative.py`: A second-order iterative ROHF program.
- `SOUHF.py`: A second-order UHF program. Uses the electronic Hessian to facilitate quadratic convergence.
- `SOUHF_iterative.py`: A second-order iterative UHF program.

Helper programs:
- `helper_HF.py`: A collection of helper classes and functions for Hartree-Fock.

Helper HF initialization:
```python
hf = helper_HF(mol, basis=None, memory=2, ndocc=None, scf_type='DF', guess='core'):
```
Input parameters:
- `mol`: A Psi4 molecule object.
- `basis`: Name of basis set to be used
- `ndocc`: Number of occupied orbitals. If `None` the number of occupied orbitals is guessed based on nuclear charge.
- `scf_type`: ERI algorithm utilized to build J and K matrices.
- `guess`: Initial orbital selection.

Helper HF methods:
- `set_Cleft`: Sets alpha orbital, automatically builds density matrix.
- `diag`: Diagonalize matrix using `S^(-1/2)` orthogonalizer.
- `build_fock`: Builds the Fock matrix using current orbitals.
- `compute_hf_energy`: Computes the HF energy using current orbitals.
- `diis_add`: Adds a matrix to the DIIS vector, uses FDS: SDF for error vectors.
- `diis_update`: Updates orbitals using the DIIS method.

### References
- General SCF: RHF, ROHF, UHF
    1. [[Szabo:1996](https://books.google.com/books?id=KQ3DAgAAQBAJ&printsec=frontcover&dq=szabo+%26+ostlund&hl=en&sa=X&ved=0ahUKEwiYhv6A8YjUAhXLSCYKHdH5AJ4Q6AEIJjAA#v=onepage&q=szabo%20%26%20ostlund&f=false)] A. Szabo and N. S. Ostlund, *Modern Quantum Chemistry: Introduction to Advanced Electronic Structure Theory.* Courier Corporation, 1996.
    2. [[Tsuchimochi:2010:141102](https://aip.scitation.org/doi/10.1063/1.3503173)] T. Tsuchimochi and G. E. Scuseria, *J. Chem. Phys.* **133**, 1411202 (2010)
    3. [[Binkley:1974:1421](https://www.tandfonline.com/doi/abs/10.1080/00268977400102701)] J. S. Binkley and J. A. Pople, *Mol. Phys.* **28**, 1423 (1974)

- Direct Inversion of the Iterative Subspace (DIIS)
    1. [[Sherrill:1998](http://vergil.chemistry.gatech.edu/notes/diis/diis.pdf)] C. D. Sherrill., "Some Comments on Accellerating Convergence of Iterative Sequences Using Direct Inversion of the Iterative Subspace," web. (1998) 
    2. [[Pulay:1969:197](https://www.tandfonline.com/doi/abs/10.1080/00268976900100941)] P. Pulay, *Mol. Phys.* **17**, 197 (1969)
    3. [[Pulay:1980:393](https://www.sciencedirect.com/science/article/pii/0009261480803964?via%3Dihub)] P. Pulay, *Chem. Phys. Lett.* **73**, 393 (1980)

- Second-Order Convergence Methods
    1. [[Helgaker:2000](https://books.google.com/books?id=lNVLBAAAQBAJ&source=gbs_navlinks_s)] T. Helgaker, P. Jorgensen, and J. Olsen, *Molecular Electronic Structure Theory.* John Wiley & Sons, Inc., 2000.

- Preconditioned Conjugate Gradient (PCG)
    1. [[Shewchuk:1994](http://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf)] J. R. Shewchuk, "An Introduction to the Conjugate Gradient Method Without the Agonizing Pain," web. (1994)

- Effective Fragment Potneitals & LIBEFP
    1. [[Kaliman:2013:2284](http://dx.doi.org/10.1002/jcc.23375)] I. Kaliman and L. Slipchenko, "LIBEFP: A New Parallel Implementation of the Effective Fragment Potential Method as a Portable Software Library." *J. Comput. Chem.*, **34**, 2284 (2013).
     
- Hartree-Fock Gradients & Hessians
    1. [[Pople:1979:225](https://onlinelibrary.wiley.com/doi/abs/10.1002/qua.560160825)] J. A. Pople, R. Krishnan, H. B. Schlegel, and J. S. Binkley. *Int. J. Quantum Chem. Symp.* **13**, 225 (1979)

