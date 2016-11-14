# A simple Psi 4 input script to compute SAPT interaction energies
#
# Created by: Daniel G. A. Smith
# Date: 12/1/14
# License: GPL v3.0

# Warning! Work in progess
#

import time
import numpy as np
from helper_SAPT import *
np.set_printoptions(precision=5, linewidth=200, threshold=2000, suppress=True)
import psi4

# Set Psi4 & NumPy memory options
psi4.core.set_memory(int(2e9), False)
psi4.set_output_file('output.dat', False)

numpy_memory = 2

# Set molecule to dimer
dimer = psi4.geometry("""
He  0  0  0
symmetry c1
""")

psi4.set_options({'basis':'aug-cc-pvqz',
                  'e_convergence':1e-8,
                  'd_convergence':1e-8})

scf_e, wfn = psi4.energy('SCF', return_wfn=True)

Co = wfn.Ca_subset("AO", "OCC")
Cv = wfn.Ca_subset("AO", "VIR")
epsilon = np.asarray(wfn.epsilon_a())

nbf = wfn.nmo()
nocc = wfn.nalpha()
nvir = nbf - nocc
nov = nocc * nvir

eps_v = epsilon[nocc:]
eps_o = epsilon[:nocc]

# Integral generation from Psi4's MintsHelper
t = time.time()
mints = psi4.core.MintsHelper(wfn.basisset())
S = np.asarray(mints.ao_overlap())
v_ijab = np.asarray(mints.mo_eri(Co, Co, Cv, Cv))
v_iajb = np.asarray(mints.mo_eri(Co, Cv, Co, Cv))
Co = np.asarray(Co)
Cv = np.asarray(Cv)

# Grab perturbation tensors in MO basis
tmp_dipoles = mints.so_dipole()
dipoles_xyz = []
for num in range(3):
    Fso = np.asarray(tmp_dipoles[num])
    Fia = (Co.T).dot(Fso).dot(Cv)
    Fia *= -2
    dipoles_xyz.append(Fia)

E1  = np.einsum('ab,ij->iajb', np.diag(eps_v), np.diag(np.ones(nocc)))
E1 -= np.einsum('ij,ab->iajb', np.diag(eps_o), np.diag(np.ones(nvir)))
E1 += 4 * v_iajb
E1 -= v_ijab.swapaxes(1, 2)
E1 -= v_iajb.swapaxes(0, 2)

C6 = np.complex(0, 0)
leg_points = 16
fdds_lambda = 0.2
print('  Omega             value            weight             sum')
for point, weight in zip(*np.polynomial.legendre.leggauss(leg_points)):
    omega = fdds_lambda * (1.0 - point) / (1.0 + point)

    #print E1
    tmp = E1.reshape(nov, nov).astype(np.complex)
    tmp[np.diag_indices_from(tmp)] -= np.complex(0, omega)
    tmp = np.linalg.inv(tmp)

    dip1 = dipoles_xyz[0].ravel().astype(np.complex)
    value = np.einsum('p,pq,q->', dip1, tmp, dip1)

    C6 += ((weight * value) ** 2) * ( (2 * fdds_lambda) / (point + 1)**2 )
    print('{:.3e}   {:.3e}   {:.3e}   {:.3e}'.format(omega, value, weight, weight*value))

C6 *= 3.0 / np.pi
C6 = C6.real - C6.imag
print('\nComputed C6: % 4.4f' % C6)
print('Limit        % 4.4f' % 1.322)
