"""
A restricted open-shell Hartree-Fock script using the Psi4NumPy Formalism

References:
- Equations and algorithm taken from Psi4
"""

__authors__ = "Daniel G. A. Smith"
__credits__ = ["Daniel G. A. Smith"]

__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__ = "BSD-3-Clause"
__date__ = "2017-9-30"

import time
import numpy as np
import helper_HF as scf_helper
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

psi4.set_options({'guess': 'gwh',
                  'basis': 'aug-cc-pvdz',
                  'scf_type': 'df',
                  'e_convergence': 1e-8,
                  'reference': 'rohf'})

wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option('BASIS'))

# Set occupations
nocca = wfn.nalpha()
noccb = wfn.nbeta()
ndocc = min(nocca, noccb)
nocc = max(nocca, noccb)
nsocc = nocc - ndocc

# Set defaults
maxiter = 20
E_conv = 1.0E-8
D_conv = 1.0E-8
guess = 'gwh'

# Integral generation from Psi4's MintsHelper
t = time.time()
mints = psi4.core.MintsHelper(wfn.basisset())
S = np.asarray(mints.ao_overlap())
nbf = S.shape[0]

print('\nNumber of doubly occupied orbitals: %d' % ndocc)
print('Number of singly occupied orbitals:   %d' % nsocc)
print('Number of basis functions:            %d' % nbf)

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

if guess == 'gwh':
    F = 0.875 * S * (np.diag(H)[:, None] + np.diag(H))
    F[np.diag_indices_from(F)] = np.diag(H)
elif guess == 'core':
    F = H.copy()
else:
    raise Exception("Unrecognized guess type %s. Please use 'core' or 'gwh'." % guess)

# Build initial orbitals and density matrices
Hp = A.dot(F).dot(A)
e, Ct = np.linalg.eigh(Hp)
C = A.dot(Ct)
Cnocc = C[:, :nocc]
Docc = np.dot(Cnocc, Cnocc.T)
Cndocc = C[:, :ndocc]
Ddocc = np.dot(Cndocc, Cndocc.T)

t = time.time()
E = 0.0
Enuc = mol.nuclear_repulsion_energy()
Eold = 0.0

# Initialize the JK object
jk = psi4.core.JK.build(wfn.basisset())
jk.initialize()

# Build a DIIS helper object
diis = scf_helper.DIIS_helper()

print('\nTotal time taken for setup: %.3f seconds' % (time.time() - t))

print('\nStart SCF iterations:\n')
t = time.time()

for SCF_ITER in range(1, maxiter + 1):

    # Build a and b fock matrices
    J, K = scf_helper.compute_jk(jk, [C[:, :nocc], C[:, :ndocc]])
    J = J[0] + J[1]
    Fa = H + J - K[0]
    Fb = H + J - K[1]

    # Build MO Fock matrix
    moFa = (C.T).dot(Fa).dot(C)
    moFb = (C.T).dot(Fb).dot(C)

    # Special note on the ROHF Fock matrix (taken from Psi4)
    # Fo = open-shell fock matrix = 0.5 Fa
    # Fc = closed-shell fock matrix = 0.5 (Fa + Fb)
    #
    # The effective Fock matrix has the following structure
    #          |  closed     open    virtual
    #  ----------------------------------------
    #  closed  |    Fc     2(Fc-Fo)    Fc
    #  open    | 2(Fc-Fo)     Fc      2Fo
    #  virtual |    Fc       2Fo       Fc

    #print moFa[ndocc:nocc, ndocc:nocc] + moFb[ndocc:nocc, ndocc:nocc]
    moFeff = 0.5 * (moFa + moFb)
    moFeff[:ndocc, ndocc:nocc] = moFb[:ndocc, ndocc:nocc]
    moFeff[ndocc:nocc, :ndocc] = moFb[ndocc:nocc, :ndocc]
    moFeff[ndocc:nocc, nocc:] = moFa[ndocc:nocc, nocc:]
    moFeff[nocc:, ndocc:nocc] = moFa[nocc:, ndocc:nocc]

    # Back transform to AO Fock
    Feff = (Ct).dot(moFeff).dot(Ct.T)

    # Build gradient
    IFock = moFeff[:nocc, ndocc:].copy()
    IFock[:, :nsocc] /= 2
    IFock[ndocc:, :] /= 2
    IFock[ndocc:, :nsocc] = 0.0
#    IFock[np.diag_indices_from(IFock)] = 0.0
    diis_e = (Ct[:, :nocc]).dot(IFock).dot(Ct[:, ndocc:].T)
    diis.add(Feff, diis_e)

    # SCF energy and update
    SCF_E  = np.einsum('pq,pq->', Docc + Ddocc, H)
    SCF_E += np.einsum('pq,pq->', Docc, Fa)
    SCF_E += np.einsum('pq,pq->', Ddocc, Fb)
    SCF_E *= 0.5
    SCF_E += Enuc

    dRMS = np.mean(diis_e**2)**0.5
    print('SCF Iteration %3d: Energy = %4.16f   dE = % 1.5E   dRMS = %1.5E'
          % (SCF_ITER, SCF_E, (SCF_E - Eold), dRMS))
    if (abs(SCF_E - Eold) < E_conv) and (dRMS < D_conv):
        break

    Eold = SCF_E

    # Build new orbitals
    Feff = diis.extrapolate()
    e, Ct = np.linalg.eigh(Feff)
    C = A.dot(Ct)

    Cnocc = C[:, :nocc]
    Docc = np.dot(Cnocc, Cnocc.T)
    Cndocc = C[:, :ndocc]
    Ddocc = np.dot(Cndocc, Cndocc.T)

    if SCF_ITER == maxiter:
        clean()
        raise Exception("Maximum number of SCF cycles exceeded.")

print('Total time for SCF iterations: %.3f seconds \n' % (time.time() - t))

print('Final SCF energy: %.8f hartree' % SCF_E)
# Compare with Psi4
SCF_E_psi = psi4.energy('SCF')
psi4.compare_values(SCF_E_psi, SCF_E, 6, 'SCF Energy')
