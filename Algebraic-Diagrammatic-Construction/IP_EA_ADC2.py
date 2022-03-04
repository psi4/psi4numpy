'''
A reference implementation of ADC(2) for the calculation of ionization
potentials and electron affinities for a restricted Hartree-Fock 
reference. A spin orbital formulation is used, as it simplifies the 
equations.

References:
    A. L. Dempwolff, M. Schneider, M. Hodecker and A. Dreuw, J. Chem. Phys., 150, 064108 (2019).
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
e_mo = np.repeat(e_mo, 2)

# Build some slices
o = slice(None, nocc)
v = slice(nocc, None)

# Calculate intermediates
e_ia = e_mo[o, None] - e_mo[None, v]
e_ija = e_mo[o, None, None] + e_ia[None]
e_iab = e_ia[:, :, None] - e_mo[None, None, v]
e_ijab = e_ija[:, :, :, None] - e_mo[None, None, None, v]
t2 = eri[o, o, v, v] / e_ijab

# Print the MP2 energy
e_mp2 = einsum('ijab,ijab->', t2, eri[o, o, v, v]) * 0.25
print('RHF total energy:       %16.10f' % e_scf)
print('MP2 correlation energy: %16.10f' % e_mp2)
print('MP2 total energy:       %16.10f' % (e_scf + e_mp2))
psi4.compare_values(psi4.energy('mp2'), e_mp2 + e_scf, 6, 'MP2 Energy')

# Construct the singles-singles (1h-1h) space (eq A5)
h_hh = np.diag(e_mo[o])
h_hh += einsum('ikab,jkab->ij', t2, eri[o, o, v, v]) * 0.25
h_hh += einsum('jkab,ikab->ij', t2, eri[o, o, v, v]) * 0.25

# Construct the single-singles (1p-1p) space (adapted from eq A5)
h_pp = np.diag(e_mo[v])
h_pp -= einsum('ijac,ijbc->ab', t2, eri[o, o, v, v]) * 0.25
h_pp -= einsum('ijbc,ijac->ab', t2, eri[o, o, v, v]) * 0.25


# Define the operation representing the dot-product of the IP-ADC(2) matrix
# with an arbitrary state vector (eq A3 & A4)
def ip_matvec(y):
    y = np.array(y, order='C')
    r = np.zeros_like(y)

    yi = y[:nocc]
    ri = r[:nocc]
    yija = y[nocc:].reshape(nocc, nocc, nvir)
    rija = r[nocc:].reshape(nocc, nocc, nvir)

    ri += np.dot(h_hh, yi)
    ri += einsum('ijak,ija->k', eri[o, o, v, o], yija) * np.sqrt(0.5)

    rija += einsum('ijak,k->ija', eri[o, o, v, o], yi) * np.sqrt(0.5)
    rija += einsum('ija,ija->ija', e_ija, yija)

    return r


# Define the operation representing the dot-product of the EA-ADC(2) matrix
# with an arbitrary state vector (adapted from eq A3 & A4)
def ea_matvec(y):
    y = np.array(y, order='C')
    r = np.zeros_like(y)

    ya = y[:nvir]
    ra = r[:nvir]
    yiab = y[nvir:].reshape(nocc, nvir, nvir)
    riab = r[nvir:].reshape(nocc, nvir, nvir)

    ra += np.dot(h_pp, ya)
    ra += einsum('abic,iab->c', eri[v, v, o, v], yiab) * np.sqrt(0.5)

    riab += einsum('abic,c->iab', eri[v, v, o, v], ya) * np.sqrt(0.5)
    riab += einsum('iab,iab->iab', -e_iab, yiab)

    return r


# Compute the diagonal of the IP-ADC(2) matrix to use as a preconditioner
# for the Davidson algorithm, and to generate the guess vectors
diag = np.concatenate([np.diag(h_hh), e_ija.ravel()])
arg = np.argsort(np.absolute(diag))
guess = np.zeros((diag.size, n_states))
for i in range(n_states):
    guess[arg[i], i] = 1.0

# Compute the IPs
e_ip, v_ip = davidson(ip_matvec, guess, diag, tol=tol)

# Print the IPs - each should be doubly degenerate
# Also printed is the quasiparticle weight (weight in the singles space)
print('\n%2s %16s %16s %16s' % ('#', 'IP (Ha)', 'IP (eV)', 'QP weight'))
for i in range(n_states):
    qpwt = np.linalg.norm(v_ip[:nocc, i])**2
    print('%2d %16.8f %16.8f %16.8f' % (i, -e_ip[i], -e_ip[i] * 27.21139664, qpwt))
print()

# Compute the diagonal of the EA-ADC(2) matrix to use as a preconditioner
# for the Davidson algorithm, and to generate the guess vectors
diag = np.concatenate([np.diag(h_pp), -e_iab.ravel()])
arg = np.argsort(np.absolute(diag))
guess = np.zeros((diag.size, n_states))
for i in range(n_states):
    guess[arg[i], i] = 1.0

# Compute the EAs
e_ea, v_ea = davidson(ea_matvec, guess, diag, tol=tol)

# Print the states - each should be doubly degenerate
# Also printed is the quasiparticle weight (weight in the singles space)
print('\n%2s %16s %16s %16s' % ('#', 'EA (Ha)', 'EA (eV)', 'QP weight'))
for i in range(n_states):
    qpwt = np.linalg.norm(v_ea[:nvir, i])**2
    print('%2d %16.8f %16.8f %16.8f' % (i, e_ea[i], e_ea[i] * 27.21139664, qpwt))
