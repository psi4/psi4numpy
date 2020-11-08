Self-Consistent Field Response (SCF Response)
=============================================

Contents:
- `CPHF.py`: Computing the Hartree--Fock static dipole polarizability via coupled-perturbed Hartree--Fock (CPHF) theory
- `TDHF.py`: Implements time-dependent Hartree--Fock via linear response theory.
- `beta.py`: Computing the first dipole hyperpolarizability beta, from an RHF reference
- `helper_CPHF.py`: Helper classes for molecular properties computed using CPHF theory.
- `vcd.py`: SCF vibrational circular dichroism (VCD) simulations.

Helper initialization:
```python
helper = helper_CPHF(molecule)
```
- `molecule` is a `psi4.core.Molecule` object
- `numpy_memory` (optional) is the number of GB allotted to the CPHF helper object

### References
1. [[Szabo:1996](https://books.google.com/books?id=KQ3DAgAAQBAJ&printsec=frontcover&dq=szabo+%26+ostlund&hl=en&sa=X&ved=0ahUKEwiYhv6A8YjUAhXLSCYKHdH5AJ4Q6AEIJjAA#v=onepage&q=szabo%20%26%20ostlund&f=false)] A. Szabo and N. S. Ostlund, *Modern Quantum Chemistry: Introduction to Advanced Electronic Structure Theory.* Courier Corporation, 1996.
2. [[Helgaker:2000](https://books.google.com/books?id=lNVLBAAAQBAJ&source=gbs_navlinks_s)] T. Helgaker, P. Jorgensen, and J. Olsen, *Molecular Electronic Structure Theory.* John Wiley & Sons, Inc., 2000.
3. [[Amos:1985:2186](https://pubs.acs.org/doi/abs/10.1021/j100257a010)] R. D. Amos, N. C. Handy, P. J. Knowles, J. E. Rice, and A. J. Stone, *J. Phys. Chem.* **89**, 2186 (1985)
4. [[Jiemchooroj:2006:124306](https://aip.scitation.org/doi/abs/10.1063/1.2348882)] A. Jiemchooroj, P. Norman, and B. E. Sernelius, *J. Chem. Phys.* **125**, 124306 (2006). 
5. [[Gauss:1991:2623](https://aip.scitation.org/doi/10.1063/1.460915)] J. Gauss, J. F. Stanton, R. J. Bartlett, *J. Chem. Phys.* **95**, 2623 (1991).
6. [[Stephens:1985:748](https://pubs.acs.org/doi/10.1021/j100251a006)] P. J. Stephens, *J. Phys. Chem.* **89**, 748 (1985).
7. [[Amos:1987:21](https://www.sciencedirect.com/science/article/pii/0009261487800465)] R. D. Amos, N. C. Handy, K. J. Jalkanen, P. J. Stephens, *Chem. Phys. Lett.* **133**, 21 (1987).
8. [[Cioslowski:1989:8333](https://pubs.acs.org/doi/10.1021/ja00204a001)] J. Cioslowski, *J. Am. Chem. Soc.* **111**, 8333 (1989).
