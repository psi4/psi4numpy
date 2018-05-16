"""
Restricted Hartree-Fock script using direct second-order convergence
acceleration.

References:
- RHF equations & algorithms from [Szabo:1996]
- SO equations & algorithm from [Helgaker:2000]
"""

__authors__ = "Daniel G. A. Smith"
__credits__ = ["Daniel G. A. Smith"]

__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__ = "BSD-3-Clause"
__date__ = "2017-9-30"

import time
import numpy as np
np.set_printoptions(precision=3, linewidth=200, suppress=True)
from helper_HF import *
import psi4

# Memory for Psi4 in GB
psi4.set_memory('500 MB')
psi4.core.set_output_file('output.dat', False)

# Memory for numpy in GB
numpy_memory = 2

mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
symmetry c1
""")

psi4.set_options({'basis': 'cc-pvdz',
                  'guess': 'sad',
                  'd_convergence': 1e-13,
                  'e_convergence': 1e-13})

# Build objects
diis = DIIS_helper()
hf = helper_HF(mol, scf_type='PK', guess='SAD')
ndocc = hf.ndocc
nvirt = hf.nvirt
mints = psi4.core.MintsHelper(hf.wfn.basisset())
mI = mints.ao_eri()
hf.diag(hf.H, set_C=True)

# Build Matrix and Numpy arrays that share memory
# Updating npC changes mC
mC = psi4.core.Matrix(hf.nbf, hf.nbf)
npC = np.asarray(mC)
occ_mC = psi4.core.Matrix(hf.nbf, hf.ndocc)
occ_npC = np.asarray(occ_mC)

# Knobs
E_conv = 1e-8
D_conv = 1e-8

print('\nStart SCF iterations:\n')
t = time.time()
E = 0.0
Eold = 0.0
Dold = 0.0
iter_type = 'DIAG'

for SCF_ITER in range(1, 20):

    # Build new fock matrix
    F = hf.build_fock()

    # DIIS error and update
    diis_e = F.dot(hf.Da).dot(hf.S) - hf.S.dot(hf.Da).dot(F)
    diis_e = (hf.A).dot(diis_e).dot(hf.A)
    diis.add(F, diis_e)

    # SCF energy and update
    scf_e = hf.compute_hf_energy()
    dRMS = np.mean(diis_e**2)**0.5
    print('SCF Iteration %3d: Energy = %4.16f   dE = % 1.3E   dRMS = %1.3E   %s' %
          (SCF_ITER, hf.scf_e, (hf.scf_e - Eold), dRMS, iter_type))
    if (abs(hf.scf_e - Eold) < E_conv) and (dRMS < D_conv):
        break

    Eold = hf.scf_e
    Dold = hf.Da

    if np.any(diis_e > 0.1):
        F = diis.extrapolate()
        e, C = hf.diag(F)
        hf.set_Cleft(C)
        iter_type = 'DIIS'
    else:
        # Build MO Fock matrix & electronic gradient
        moF = np.einsum('ui,vj,uv', hf.Ca, hf.Ca, F)
        gn = -4 * moF[:ndocc, ndocc:] # [Helgaker:2000] Eqn. 10.8.34, pp. 484

        # AO -> MO ERI transform
        # Only transform occupied on first index, oN^4 <<< N^5
        # Update the Psi4 Matrices that these numpy arrays point to
        npC[:] = hf.Ca
        occ_npC[:] = hf.Ca[:, :ndocc]
        MO = np.asarray(mints.mo_transform(mI, occ_mC, mC, mC, mC))

        # Build electronic Hessian: [Helgaker:2000] Eqn. 10.9.22, pp. 494
        # Biajb = 4 * [(Fab - Fij) + 4 * (ia|jb) - (ib|ja) - (ij|ab)]
        Biajb = np.einsum('ab,ij->iajb', moF[ndocc:, ndocc:], np.diag(np.ones(ndocc)))
        Biajb -= np.einsum('ij,ab->iajb', moF[:ndocc:, :ndocc], np.diag(np.ones(nvirt)))
        Biajb += 4 * MO[:, ndocc:, :ndocc, ndocc:]
        Biajb -= MO[:, ndocc:, :ndocc, ndocc:].swapaxes(0, 2)
        Biajb -= MO[:, :ndocc, ndocc:, ndocc:].swapaxes(1, 2)
        Biajb *= 4

        # Invert B, (o^3 v^3); solves Newton equations H*x = B
        Binv = np.linalg.inv(Biajb.reshape(ndocc * nvirt, -1)).reshape(ndocc, nvirt, ndocc, nvirt)

        # Build orbital rotation matrix from Hessian inverse & gradient
        x = np.einsum('iajb,ia->jb', Binv, gn)
        U = np.zeros_like(hf.Ca)
        U[:ndocc, ndocc:] = x
        U[ndocc:, :ndocc] = -x.T
        U += 0.5 * np.dot(U, U)
        U[np.diag_indices_from(hf.A)] += 1

        # Easy access to Schmidt orthogonalization
        U, r = np.linalg.qr(U.T)

        # Rotate and set orbitals
        C = hf.Ca.dot(U)
        hf.set_Cleft(C)
        iter_type = 'SOSCF'

print('Total time taken for SCF iterations: %.3f seconds \n' % (time.time() - t))

print('Final SCF energy:     %.8f hartree' % hf.scf_e)

# Compare to Psi4
SCF_E_psi = psi4.energy('SCF')
psi4.compare_values(SCF_E_psi, hf.scf_e, 6, 'SCF Energy')

