"""
A reference implementation of the Hartree-Fock static dipole
polarizability.

References:
- Equations and algorithms from [Szabo:1996] and Project 3 from
Daniel Crawford's programming website:
http://github.com/CrawfordGroup/ProgrammingProjects
"""

__authors__   =  "Daniel G. A. Smith"
__credits__   = ["Daniel G. A. Smith", "Dominic A. Sirianni", "Eric J. Berquist",
"Robert M. Parrish"]

__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__   = "BSD-3-Clause"
__date__      = "2017-05-23"

import time
import numpy as np
np.set_printoptions(precision=5, linewidth=200, suppress=True)
import psi4

import os.path
import sys
dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dirname, '../../Self-Consistent-Field'))
from helper_HF import DIIS_helper

# Memory for Psi4 in GB
psi4.set_memory('2 GB')
psi4.core.set_output_file("output.dat", False)

mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
symmetry c1
""")

# Set options for CPHF
psi4.set_options({"basis": "aug-cc-pVDZ",
                  "scf_type": "direct",
                  "df_scf_guess": False,
                  "e_convergence": 1e-9,
                  "d_convergence": 1e-9,
                  "cphf_tasks": ['polarizability']})

# Set defaults
# Can be direct or iterative
method = 'iterative'
numpy_memory = 2
use_diis = True

# Iterative settings
maxiter = 20
conv = 1.e-9

# Compute the reference wavefunction and CPHF using Psi
scf_e, scf_wfn = psi4.energy('SCF', return_wfn=True)

C = scf_wfn.Ca()
Co = scf_wfn.Ca_subset("AO", "OCC")
Cv = scf_wfn.Ca_subset("AO", "VIR")
epsilon = np.asarray(scf_wfn.epsilon_a())

nbf = scf_wfn.nmo()
nocc = scf_wfn.nalpha()
nvir = nbf - nocc

# Integral generation from Psi4's MintsHelper
t = time.time()
mints = psi4.core.MintsHelper(scf_wfn.basisset())
S = np.asarray(mints.ao_overlap())

# Get nbf and ndocc for closed shell molecules
print('\nNumber of occupied orbitals: %d' % nocc)
print('Number of basis functions: %d' % nbf)

# Grab perturbation tensors in MO basis
nCo = np.asarray(Co)
nCv = np.asarray(Cv)
tmp_dipoles = mints.so_dipole()
dipoles_xyz = []
for num in range(3):
    Fso = np.asarray(tmp_dipoles[num])
    Fia = (nCo.T).dot(Fso).dot(nCv)
    Fia *= -2
    dipoles_xyz.append(Fia)

if method == 'direct':
    # Run a quick check to make sure everything will fit into memory
    I_Size = (nbf ** 4) * 8.e-9
    oNNN_Size = (nocc * nbf ** 3) * 8.e-9
    ovov_Size = (nocc * nocc * nvir * nvir) * 8.e-9
    print("\nTensor sizes:")
    print("ERI tensor           %4.2f GB." % I_Size)
    print("oNNN MO tensor       %4.2f GB." % oNNN_Size)
    print("ovov Hessian tensor  %4.2f GB." % ovov_Size)

    # Estimate memory usage
    memory_footprint = I_Size * 1.5
    if I_Size > numpy_memory:
        psi4.core.clean()
        raise Exception("Estimated memory utilization (%4.2f GB) exceeds numpy_memory \
                        limit of %4.2f GB." % (memory_footprint, numpy_memory))

    # Compute electronic hessian
    print('\nForming hessian...')
    t = time.time()
    docc = np.diag(np.ones(nocc))
    dvir = np.diag(np.ones(nvir))
    eps_diag = epsilon[nocc:].reshape(-1, 1) - epsilon[:nocc]

    # Form oNNN MO tensor, oN^4 cost
    MO = np.asarray(mints.mo_eri(Co, C, C, C))

    H = np.einsum('ai,ij,ab->iajb', eps_diag, docc, dvir)
    H += 4 * MO[:, nocc:, :nocc, nocc:]
    H -= MO[:, nocc:, :nocc, nocc:].swapaxes(0, 2)
    H -= MO[:, :nocc, nocc:, nocc:].swapaxes(1, 2)

    print('...formed hessian in %.3f seconds.' % (time.time() - t))

    # Invert hessian (o^3 v^3)
    print('\nInverting hessian...')
    t = time.time()
    Hinv = np.linalg.inv(H.reshape(nocc * nvir, -1)).reshape(nocc, nvir, nocc, nvir)
    print('...inverted hessian in %.3f seconds.' % (time.time() - t))

    # Form perturbation response vector for each dipole component
    x = []
    for numx in range(3):
        xcomp = np.einsum('iajb,ia->jb', Hinv, dipoles_xyz[numx])
        x.append(xcomp)


elif method == 'iterative':

    # Init JK object
    jk = psi4.core.JK.build(scf_wfn.basisset())
    jk.initialize()

    # Add blank matrices to the jk object and numpy hooks to C_right
    npC_right = []
    for xyz in range(3):
        jk.C_left_add(Co)
        mC = psi4.core.Matrix(nbf, nocc)
        npC_right.append(np.asarray(mC))
        jk.C_right_add(mC)

    # Build initial guess, previous vectors, diis object, and C_left updates
    x = []
    x_old = []
    diis = []
    ia_denom = - epsilon[:nocc].reshape(-1, 1) + epsilon[nocc:]
    for xyz in range(3):
        x.append(dipoles_xyz[xyz] / ia_denom)
        x_old.append(np.zeros(ia_denom.shape))
        diis.append(DIIS_helper())

    # Convert Co and Cv to numpy arrays
    mCo = Co
    Co = np.asarray(Co)
    Cv = np.asarray(Cv)

    print('\nStarting CPHF iterations:')
    t = time.time()
    for CPHF_ITER in range(1, maxiter + 1):

        # Update jk's C_right
        for xyz in range(3):
            npC_right[xyz][:] = Cv.dot(x[xyz].T)

        # Compute JK objects
        jk.compute()

        # Update amplitudes
        for xyz in range(3):
            # Build J and K objects
            J = np.asarray(jk.J()[xyz])
            K = np.asarray(jk.K()[xyz])

            # Bulid new guess
            X = dipoles_xyz[xyz].copy()
            X -= (Co.T).dot(4 * J - K.T - K).dot(Cv)
            X /= ia_denom

            # DIIS for good measure
            if use_diis:
                diis[xyz].add(X, X - x_old[xyz])
                X = diis[xyz].extrapolate()
            x[xyz] = X.copy()

        # Check for convergence
        rms = []
        for xyz in range(3):
            rms.append(np.max((x[xyz] - x_old[xyz]) ** 2))
            x_old[xyz] = x[xyz]

        avg_RMS = sum(rms) / 3
        max_RMS = max(rms)

        if max_RMS < conv:
            print('CPHF converged in %d iterations and %.2f seconds.' % (CPHF_ITER, time.time() - t))
            break

        print('CPHF Iteration %3d: Average RMS = %3.8f  Maximum RMS = %3.8f' %
                (CPHF_ITER, avg_RMS, max_RMS))


else:
    raise Exception("Method %s is not recognized" % method)


# Compute 3x3 polarizability tensor
polar = np.empty((3, 3))
for numx in range(3):
    for numf in range(3):
        polar[numx, numf] = np.einsum('ia,ia->', x[numx], dipoles_xyz[numf])

# Compare against reference
ref = np.array([
    [8.01522720,  0.00000000,  0.00000000],
    [0.00000000, 12.50372724,  0.00000000],
    [0.00000000,  0.00000000, 10.04226990]
])
assert np.allclose(polar, ref, rtol=0, atol=1.e-3)

print('\nCPHF Dipole Polarizability:')
print(np.around(polar, 5))
