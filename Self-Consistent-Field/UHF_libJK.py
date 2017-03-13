# A simple Psi 4 input script to compute a SCF reference using Psi4's libJK
# Requires numpy 1.7.2+
#
# Created by: Daniel G. A. Smith
# Date: 4/1/15
# License: GPL v3.0
#

import time
import numpy as np
from helper_HF import *
np.set_printoptions(precision=5, linewidth=200, suppress=True)
import psi4

# Memory for Psi4 in GB
psi4.set_memory('2 GB')
psi4.core.set_output_file('output.dat', False)

# Memory for numpy in GB
numpy_memory = 2

# Triplet O2
mol = psi4.geometry("""
    0 3
    O
    O 1 1.2
symmetry c1
""")

psi4.set_options({'guess':      'core',
                  'basis':      'aug-cc-pvdz',
                  'scf_type':   'df',
                  'e_convergence': 1e-8,
                  'reference':  'uhf'})


# Set defaults
maxiter = 40
E_conv = 1.0E-8
D_conv = 1.0E-5

# Integral generation from Psi4's MintsHelper
t = time.time()
wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option('BASIS'))
mints = psi4.core.MintsHelper(wfn.basisset())
S = np.asarray(mints.ao_overlap())

# Get nbf and ndocc for closed shell molecules
nbf = wfn.nso()
nalpha = wfn.nalpha()
nbeta = wfn.nbeta()

print('\nNumber of doubly occupied orbitals: %d' % nbeta)
print('Number of singly occupied orbitals: %d' % (nalpha - nbeta))
print('Number of basis functions: %d' % nbf)

V = np.asarray(mints.ao_potential())
T = np.asarray(mints.ao_kinetic())

print('\nTotal time taken for integrals: %.3f seconds.' % (time.time()-t))

t = time.time()

# Build H_core
H = T + V

# Orthogonalizer A = S^(-1/2)
A = mints.ao_overlap()
A.power(-0.5, 1.e-16)
A = np.asarray(A)


def diag_H(H, nocc):
    Hp = A.dot(H).dot(A)
    e, C2 = np.linalg.eigh(Hp)
    C = A.dot(C2)
    Cocc = C[:, :nocc]
    D = np.einsum('pi,qi->pq', Cocc, Cocc)
    return (C, D)
    
Ca, Da = diag_H(H, nalpha)    
Cb, Db = diag_H(H, nbeta)    

t = time.time()
E = 0.0
Enuc = mol.nuclear_repulsion_energy()
Eold = 0.0

Fock_list = []
DIIS_error = []

# Build a C matrix and share data with the numpy array npC
Cocca = psi4.core.Matrix(nbf, nalpha)
npCa = np.asarray(Cocca)
npCa[:] = Ca[:, :nalpha]

Coccb = psi4.core.Matrix(nbf, nbeta)
npCb = np.asarray(Coccb)
npCb[:] = Cb[:, :nbeta]

# Initialize the JK object
jk = psi4.core.JK.build(wfn.basisset())
jk.initialize()
jk.C_left_add(Cocca)
jk.C_left_add(Coccb)

# Build a DIIS helper object
diisa = DIIS_helper()
diisb = DIIS_helper()

print('\nTotal time taken for setup: %.3f seconds' % (time.time() - t))

print('\nStart SCF iterations:\n')
t = time.time()

for SCF_ITER in range(1, maxiter + 1):

    npCa[:] = Ca[:, :nalpha]
    npCb[:] = Cb[:, :nbeta]
    jk.compute()

    # Build fock matrix
    Ja = np.asarray(jk.J()[0])
    Jb = np.asarray(jk.J()[1])
    Ka = np.asarray(jk.K()[0])
    Kb = np.asarray(jk.K()[1])
    Fa = H + (Ja + Jb) - Ka
    Fb = H + (Ja + Jb) - Kb

    # DIIS error build and update
    diisa_e = Fa.dot(Da).dot(S) - S.dot(Da).dot(Fa)
    diisa_e = (A.T).dot(diisa_e).dot(A)
    diisa.add(Fa, diisa_e)

    diisb_e = Fb.dot(Db).dot(S) - S.dot(Db).dot(Fb)
    diisb_e = (A.T).dot(diisb_e).dot(A)
    diisb.add(Fb, diisb_e)

    # SCF energy and update
    SCF_E  = np.einsum('pq,pq->', Da + Db, H)
    SCF_E += np.einsum('pq,pq->', Da, Fa)
    SCF_E += np.einsum('pq,pq->', Db, Fb)
    SCF_E *= 0.5
    SCF_E += Enuc 

    dRMS = 0.5 * (np.mean(diisa_e**2)**0.5 + np.mean(diisb_e**2)**0.5)
    print('SCF Iteration %3d: Energy = %4.16f   dE = % 1.5E   dRMS = %1.5E'
          % (SCF_ITER, SCF_E, (SCF_E - Eold), dRMS))
    if (abs(SCF_E - Eold) < E_conv) and (dRMS < D_conv):
        break

    Eold = SCF_E

    Fa = diisa.extrapolate()
    Fb = diisb.extrapolate()

    # Diagonalize Fock matrix
    Ca, Da = diag_H(Fa, nalpha)    
    Cb, Db = diag_H(Fb, nbeta)    

    if SCF_ITER == maxiter:
        clean()
        raise Exception("Maximum number of SCF cycles exceeded.")

print('Total time for SCF iterations: %.3f seconds \n' % (time.time() - t))

spin_mat = (Cb[:, :nbeta].T).dot(S).dot(Ca[:, :nalpha])
spin_contam = min(nalpha, nbeta) - np.vdot(spin_mat, spin_mat)
print('Spin Contamination Metric: %1.5E\n' % spin_contam)

print('Final SCF energy: %.8f hartree' % SCF_E)

# Compare to Psi4
SCF_E_psi = psi4.energy('SCF')
psi4.driver.p4util.compare_values(SCF_E_psi, SCF_E, 6, 'SCF Energy')
