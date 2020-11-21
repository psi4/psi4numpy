'''
A reference implementation of ADC(2) for the calculation of excited
states for a restricted Hartree-Fock reference. A spin orbital
formulation is used, as it simplifies the equations.

References:
    M. Hodecker, PhD Thesis, Heidelberg University (2020): https://doi.org/10.11588/heidok.00028275
'''

__authors__ = 'Oliver J. Backhouse'
__credits__ = ['Oliver J. Backhouse']

__copyright__ = '(c) 2014-2020, The Psi4NumPy Developers'
__license__ = 'BSD-3-Clause'
__date__ = '2018-03-01'

import time
import numpy as np
import psi4
import functools
from adc_helper import davidson

einsum = functools.partial(np.einsum, optimize=True)

# Settings
n_states = 5
tol = 1e-8

# Set the memory and output file
psi4.set_memory('2 GB')
psi4.core.set_output_file('output.dat', False)

# Set molecule and basis
mol = psi4.geometry('''
O
H 1 1.1
H 1 1.1 2 104
symmetry c1
''')

psi4.set_options({
    'basis': '6-31g',
    'scf_type': 'pk',
    'mp2_type': 'conv',
    'e_convergence': 1e-10,
    'd_convergence': 1e-10,
    'freeze_core': 'false'
})

# Perform SCF
print('\nPerforming SCF...')
e_scf, wfn = psi4.energy('SCF', return_wfn=True)
mints = psi4.core.MintsHelper(wfn.basisset())

# Get data from the wavefunction
nocc = wfn.doccpi()[0]
nmo = wfn.nmo()
e_mo = wfn.epsilon_a().np
c_mo = wfn.Ca()

# Get the antisymmetrized spin-orbital integrals (physicist notation)
print('Building integrals...')
eri = mints.mo_spin_eri(c_mo, c_mo).np

# Expand to spin orbitals
nso = nmo * 2
nocc = nocc * 2
nvir = nso - nocc
nia = nocc * nvir
e_mo = np.repeat(e_mo, 2)

# Build some slices
o = slice(None, nocc)
v = slice(nocc, None)

# Calculate intermediates
e_ia = e_mo[o, None] - e_mo[None, v]
e_ijab = e_ia[:, None, :, None] + e_ia[None, :, None, :]
t2 = eri[o, o, v, v] / e_ijab

# Print the MP2 energy
e_mp2 = einsum('ijab,ijab->', t2, eri[o, o, v, v]) * 0.25
print('RHF total energy:       %16.10f' % e_scf)
print('MP2 correlation energy: %16.10f' % e_mp2)
print('MP2 total energy:       %16.10f' % (e_scf + e_mp2))
psi4.compare_values(psi4.energy('mp2'), e_mp2 + e_scf, 6, 'MP2 Energy')


# Define the operation representing the dot-product of the EE-ADC(2) matrix
# with an arbitrary state vector.
def matvec(y):
    y = np.array(y, order='C')
    r = np.zeros_like(y)

    yia = y[:nia].reshape(nocc, nvir)
    ria = r[:nia].reshape(nocc, nvir)
    yiajb = y[nia:].reshape(nocc, nvir, nocc, nvir)
    riajb = r[nia:].reshape(nocc, nvir, nocc, nvir)

    ria -= einsum('ia,ia->ia', e_ia, yia)
    ria -= einsum('ajbi,jb->ia', eri[v, o, v, o], yia)

    ria += einsum('acik,jkbc,jb->ia', eri[v, v, o, o], t2, yia) * 0.5
    ria += einsum('jkbc,ikac,jb->ia', eri[o, o, v, v], t2, yia) * 0.5

    tmp = einsum('cdik,jkcd->ij', eri[v, v, o, o], t2)
    tmp += einsum('jkcd,ikcd->ij', eri[o, o, v, v], t2)
    ria += einsum('ij,ja->ia', tmp, yia) * -0.25

    tmp = einsum('ackl,klbc->ab', eri[v, v, o, o], t2)
    tmp += einsum('klbc,klac->ab', eri[o, o, v, v], t2)
    ria += einsum('ab,ib->ia', tmp, yia) * -0.25

    ria += einsum('klid,kcld->ic', eri[o, o, o, v], yiajb) * 0.5
    ria += einsum('klic,kcld->id', -eri[o, o, o, v], yiajb) * 0.5
    ria += einsum('alcd,kcld->ka', -eri[v, o, v, v], yiajb) * 0.5
    ria += einsum('akcd,kcld->la', eri[v, o, v, v], yiajb) * 0.5

    riajb += einsum('kbij,ka->iajb', eri[o, v, o, o], yia) * 0.5
    riajb += einsum('kaij,kb->iajb', -eri[o, v, o, o], yia) * 0.5
    riajb += einsum('abcj,ic->iajb', -eri[v, v, v, o], yia) * 0.5
    riajb += einsum('abci,jc->iajb', eri[v, v, v, o], yia) * 0.5

    riajb += einsum('ijab,iajb->iajb', -e_ijab, yiajb)

    return r


# Compute the diagonal of the EE-ADC(2) matrix to use as a preconditioner
# for the Davidson algorithm, and to generate the guess vectors
diag = -np.concatenate([e_ia.ravel(), e_ijab.swapaxes(1, 2).ravel()])
d_ia = diag[:nia].reshape(nocc, nvir)
d_ia -= einsum('aiai->ia', eri[v, o, v, o])
d_ia += einsum('acik,ikac->ia', eri[v, v, o, o], t2) * 0.5
d_ia += einsum('ikac,ikac->ia', eri[o, o, v, v], t2) * 0.5
d_ia -= einsum('cdik,ikcd->i', eri[v, v, o, o], t2)[:, None] * 0.25
d_ia -= einsum('ikcd,ikcd->i', eri[o, o, v, v], t2)[:, None] * 0.25
d_ia -= einsum('ackl,klac->a', eri[v, v, o, o], t2)[None, :] * 0.25
d_ia -= einsum('klac,klac->a', eri[o, o, v, v], t2)[None, :] * 0.25

arg = np.argsort(np.absolute(diag))
guess = np.eye(diag.size)[:, arg[:n_states]]

# Computes the EEs
e_ee, v_ee = davidson(matvec, guess, diag, tol=tol)

# Print the excited states - eech should be either singly or triply degenerate
print('\n%2s %16s %16s' % ('#', 'EE (Ha)', 'EE (eV)'))
for i in range(n_states):
    print('%2d %16.8f %16.8f' % (i, e_ee[i], e_ee[i] * 27.21139664))
