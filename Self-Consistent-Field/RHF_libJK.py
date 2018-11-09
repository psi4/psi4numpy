"""
A restricted Hartree-Fock code using the Psi4 JK class for the 
4-index electron repulsion integrals.

References:
- Algorithms from [Szabo:1996], [Sherrill:1998], and [Pulay:1980:393]
"""

__authors__ = "Daniel G. A. Smith"
__credits__ = ["Daniel G. A. Smith"]

__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__ = "BSD-3-Clause"
__date__ = "2017-9-30"

import time
import numpy as np
np.set_printoptions(precision=5, linewidth=200, suppress=True)
import psi4
import helper_HF

# Memory & Output File
psi4.set_memory('2 GB')
psi4.core.set_output_file('output.dat', False)

# Benzene
mol = psi4.geometry("""
C  0.000  1.396  0.000
C  1.209  0.698  0.000
C  1.209 -0.698  0.000
C  0.000 -1.396  0.000
C -1.209 -0.698  0.000
C -1.209  0.698  0.000
H  0.000  2.479  0.000
H  2.147  1.240  0.000
H  2.147 -1.240  0.000
H  0.000 -2.479  0.000
H -2.147 -1.240  0.000
H -2.147  1.240  0.000
symmetry c1
""")

# Set a few options
psi4.set_options({"BASIS": "AUG-CC-PVDZ",
                  "SCF_TYPE": "DF",
                  "E_CONVERGENCE": 1.e-8})

# Set tolerances
maxiter = 12
E_conv = 1.0E-6
D_conv = 1.0E-5

# Integral generation from Psi4's MintsHelper
wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option('BASIS'))
mints = psi4.core.MintsHelper(wfn.basisset())
S = mints.ao_overlap()

# Get nbf and ndocc for closed shell molecules
nbf = wfn.nso()
ndocc = wfn.nalpha()
if wfn.nalpha() != wfn.nbeta():
    raise PsiException("Only valid for RHF wavefunctions!")

print('\nNumber of occupied orbitals: %d\n' % ndocc)
print('Number of basis functions:   %d\n' % nbf)

# Build H_core
V = mints.ao_potential()
T = mints.ao_kinetic()
H = T.clone()
H.add(V)

# Orthogonalizer A = S^(-1/2)
A = mints.ao_overlap()
A.power(-0.5, 1.e-16)

# Build diis
diis = helper_HF.DIIS_helper(max_vec=6)

# Diagonalize routine
def build_orbitals(diag):
    Fp = psi4.core.Matrix.triplet(A, diag, A, True, False, True)

    Cp = psi4.core.Matrix(nbf, nbf)
    eigvals = psi4.core.Vector(nbf)
    Fp.diagonalize(Cp, eigvals, psi4.core.DiagonalizeOrder.Ascending)

    C = psi4.core.Matrix.doublet(A, Cp, False, False)

    Cocc = psi4.core.Matrix(nbf, ndocc)
    Cocc.np[:] = C.np[:, :ndocc]

    D = psi4.core.Matrix.doublet(Cocc, Cocc, False, True)
    return C, Cocc, D

# Build core orbitals
C, Cocc, D = build_orbitals(H)

# Setup data for DIIS
t = time.time()
E = 0.0
Enuc = mol.nuclear_repulsion_energy()
Eold = 0.0
Dold = psi4.core.Matrix(nbf, nbf)
Fock_list = []
DIIS_error = []

# Initialize the JK object
jk = psi4.core.JK.build(wfn.basisset())
jk.set_memory(int(1.25e8))  # 1GB
jk.initialize()
jk.print_header()

print('\nTotal time taken for setup: %.3f seconds\n' % (time.time() - t))

print('\nStart SCF iterations:\n\n')
t = time.time()

for SCF_ITER in range(1, maxiter + 1):

    # Compute JK
    jk.C_left_add(Cocc)
    jk.compute()
    jk.C_clear()

    # Build Fock matrix
    F = H.clone()
    F.axpy(2.0, jk.J()[0])
    F.axpy(-1.0, jk.K()[0])

    # DIIS error build and update
    diis_e = psi4.core.Matrix.triplet(F, D, S, False, False, False)
    diis_e.subtract(psi4.core.Matrix.triplet(S, D, F, False, False, False))
    diis_e = psi4.core.Matrix.triplet(A, diis_e, A, False, False, False)

    diis.add(F, diis_e)

    # SCF energy and update
    FH = F.clone()
    FH.add(H)
    SCF_E = FH.vector_dot(D) + Enuc

    dRMS = diis_e.rms()

    print('SCF Iteration %3d: Energy = %4.16f   dE = % 1.5E   dRMS = %1.5E'
          % (SCF_ITER, SCF_E, (SCF_E - Eold), dRMS))
    if (abs(SCF_E - Eold) < E_conv) and (dRMS < D_conv):
        break

    Eold = SCF_E
    Dold = D

    F = psi4.core.Matrix.from_array(diis.extrapolate())

    # Diagonalize Fock matrix
    C, Cocc, D = build_orbitals(F)

    if SCF_ITER == maxiter:
        psi4.clean()
        raise Exception("Maximum number of SCF cycles exceeded.\n")

print('Total time for SCF iterations: %.3f seconds \n\n' % (time.time() - t))

print('Final SCF energy: %.8f hartree\n' % SCF_E)
psi4.compare_values(-230.7277181465556453, SCF_E, 6, 'SCF Energy')
