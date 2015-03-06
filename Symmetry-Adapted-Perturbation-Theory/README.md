Symmetry-Adapted-Perturbation-Theory
====================================
The primary operator in SAPT is the intermolecular interaction operator `\tilde{V}`, that is defined as follows:

![TildeV](../media/SAPT_V_TILDE.png)

As all SAPT quantities will make use of this expression, a SAPT helper object that can automatically build this value greatly simplifies many routine SAPT tasks. In psi4numpy this helper object can be initialized as follow:

```python
sapt = helper_SAPT(psi4, energy, dimer, memory=8)
```

As the helper is external to the main psithon input we must pass both `psi4` and `energy` global objects from the input file.
The `psi4` object contains many useful tools such as MintsHelper while the `energy` is the main energy driver for psi4 (needed for RHF here). The `dimer` is a psi4 moelcule object created with the standard `mol` tag in psithon. In addition, we set the memory limit to 8GB (default 2GB).

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
