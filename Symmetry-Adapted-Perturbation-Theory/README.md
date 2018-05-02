Symmetry-Adapted-Perturbation-Theory
====================================

The following codes are available:
- `SAPT0.py`: A simple Psi 4 input script to compute SAPT interaction energies.
- `SAPT0_ROHF.py`: A SAPT0(ROHF) script for the oxygen dimer (two triplets making a quintet).
- `SAPT0_no_S2.py`: A script to compute the SAPT0 interaction energy without the the Single-Exchange Approximation.
- `SAPT0ao.py`: A Psi 4 input script to compute SAPT interaction energies in atomic orbitals.

Helper programs:
- `helper_SAPT.py`: A collection of helper classes and functions for SAPT.

The most important operator in SAPT is the intermolecular interaction operator `\tilde{V}`, that is defined as follows:

![TildeV](../media/latex/SAPT_V_TILDE.png)

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
