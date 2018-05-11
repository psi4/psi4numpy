Coupled-Cluster Theory
======================

Reference implementations for various truncations of spin-orbital--based
coupled cluster (CC) theory based upon the direct-product decomposition (DPD)
formulation.  

The following codes are available:
- `CCSD.py`: An implementation of coupled cluster theory utilizing single and
double substitutions (CCSD)
- `CCSD_DIIS.py`: A CCSD implementation using DIIS convergence acceleration
for T1 & T2 amplitudes
- `CCSD_T.py`: An implementation of CCSD with a perturbative triples correction, 
CCSD(T)
- `TD-CCSD.py`: Explicitly time-dependent linear absorption spectra computed
within the equation-of-motion coupled cluster framework (TD-EOM-CCSD)

Helper CC initialization:
```python
ccsd = helper_CCSD(mol, freeze_core=False, memory=2)
```
Input Parameters:
- `mol`: A `psi4.core.Molecule` object
- `freeze_core`: Boolean flag to indicate the presence of frozen core orbitals
- `memory`: The allotted memory for the helper object

### References:
- Direct Product Decomposition formulation of Coupled Cluster theory:
    1. [[Stanton:1991:4334](https://aip.scitation.org/doi/10.1063/1.460620)] J. F. Stanton, J. Gauss, J. Watts, and R. J. Bartlett, *J. Chem. Phys.* **94**, 4334 (1991)

- Similarity-transformed Hamiltonian & LCCSD equations:
    1. [[Gauss:1995:3561](https://aip.scitation.org/doi/10.1063/1.470240)] J. Gauss and J. F. Stanton, *J. Chem. Phys.* **103**, 3561 (1995)

- Time-dependent Equation of Motion Coupled Cluster Singles and Doubles (TD-EOM-CCSD)
    1. [[Nascimento:2016:5834](https://pubs.acs.org/doi/abs/10.1021/acs.jctc.6b00796)] D. R. Nascimento and A. E. Deprince III, *J. Chem. Theory Comput.* **12**, 5834 (2016)

- Runge-Kutta integration:
    1. [[Cheever:2015](http://lpsa.swarthmore.edu/NumInt/NumIntIntro.html)] E. Cheever, "Approximation of Differential Equations by Numerical Integration," web. (2015)
