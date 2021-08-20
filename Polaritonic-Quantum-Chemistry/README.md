Polaritonic-Quantum-Chemistry
====================================

The following codes are available:
- `CQED_RHF.py`: A script to compute CQED-RHF ground-state of the water molecule strongly coupled to a single photon.
- `CS_CQED_CIS.py`: A script to compute the CS-CQED-CIS excitation energies of a lone MgH+ molecule (Case 1), MgH+ strongly 
                    coupled to a single photon that is resonant with the first dipole-allowed transition (Case 2), and
                    MgH+ coupled to a single photon with finite lifetime / complex energy with central energy resonant with the first dipole
                    allowed-transition (Case 3)

Helper programs:
- `helper_CQED_RHF.py`: A helper function to perform Restricted Hartree-Fock theory for the mean-field ground state of the  Pauli-Fierz Hamiltonian that includes an ab initio electronic Hamiltonian with dipolar coupling to a quantized photon mode and a quadratic self polarization energy contribution
  `helper_CS_CQED_CIS.py`: A helper function to perform configuration interaction singles for the mean-field excited-states of the Pauli-Fierz Hamiltonian including an ab initio electronic Hamiltonian with dipolar coupling to a quantized photon mode and a quadratic self polarization energy contribution

!!! NEEDS UPDATING BELOW
At the heart of CQED-RHF is an augmented Fock operator `F` that includes dipolar and quadrupolar coupling between 
the molecular electronic degrees of freedom and the photonic degree of freedom:

![CQED_RHF_FO](../media/latex/CQED_RHF_FO.png)

Where the Core Hamiltonian `H` is defined as follows:

![CQED_RHF_FO_1E](../media/latex/CQED_RHF_FO_1E.png)

and the 2-electron contributions are augmented as follows

![CQED_RHF_FO_2E](../media/latex/CQED_RHF_FO_2E.png)

As all SAPT quantities will make use of this expression, a SAPT helper object that can automatically build this value greatly simplifies many routine SAPT tasks. In psi4numpy this helper object can be initialized as follow:

```python
sapt = helper_SAPT(dimer, memory=8)
```

Where the dimer object is a Psi4 Molecule with exactly two fragments.

Indices used:
- a - occupied indices of monomer A
- b - occupied indices of monomer B
- r - virtual indices of monomer A
- s - virtual indices of monomer B

The `sapt` object now contains all of the information that we need to compute the following:
- `v`: Two electron repulsion integrals in the MO basis (e.g. `sapt.v('arar')`)
- `s`: Overlap integrals in the MO basis (e.g. `sapt.s('ab')`)
- `eps`: SCF eigenvalues (e.g. `sapt.eps('a')`)
- `potential`: Electrostatic potential of a monomer (e.g. `sapt.potential('bb', 'A')`)
- `vt`: Intermolecular interaction operator \tilde{V} (e.g. `sapt.vt('arar')`)
- `chf`: Computes CPHF orbitals for each monomer (e.g. `chf('A')`)

Where for all 4-index quantities the order of the indices is as follows:

`\tilde{V}_{0, 1}^{2, 3}`

For example, the expression for first order electrostatics is as follows:

`4 * \tilde{V}_{a, b}^{a, b}`

Computation of the \tilde{V} operator can be accomplished as:

`vt_abab = sapt.vt('abab')`

`vt_abab` is a 4-index tensor that needs to be summed over following einsum convention:

`Elst10 = 4 * np.einsum('abab', vt_abab)`

By first building a tool to compute arbitrary `\tilde{V}` tensors we can, in principle, compute up to third order SAPT easily.

### References

1. Pauli-Fierz Hamiltonian and CQED-RHF Equations
    - [[Haugland:2020:041043](https://journals.aps.org/prx/pdf/10.1103/PhysRevX.10.041043)] T. S. Haughland, E. Ronco, E. F. Kjonstad, A. Rubio, H. Koch, *Phys. Rev. X*, **10**, 041043 (2020) 
    - [[DePrince:2021:094112]](https://aip.scitation.org/doi/10.1063/5.0038748) A. E. DePrince III, *J. Chem. Phys.* **154**, 094113 (2021).
2. Detailed CQED-RHF and CQED-CIS equations and overview of algorithm   
    - [[McTague:2021:254]()] J. McTague, J. J. Foley IV, 
