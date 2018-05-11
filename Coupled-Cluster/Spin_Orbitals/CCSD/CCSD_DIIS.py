"""
Script to compute CCSD correlation energy from a RHF reference,
utilizing DIIS convergence acceleration for CCSD amplitude iterations.

References:
- CCSD algorithms from Daniel Crawford's programming website:
http://github.com/CrawfordGroup/ProgrammingProjects
- DPD formulation of CCSD equations from [Stanton:1991:4334]
- DIIS equations & algorithms from [Sherrill:1998], [Pulay:1980:393], & [Pulay:1969:197]
"""
__authors__   =  "Daniel G. A. Smith"
__credits__   =  ["Daniel G. A. Smith", "Lori A. Burns"]

__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__   = "BSD-3-Clause"
__date__      = "2014-07-29"

import time
import numpy as np
from helper_CC import *
np.set_printoptions(precision=5, linewidth=200, suppress=True)
import psi4

psi4.set_memory('2 GB')
psi4.core.set_output_file('output.dat', False)

mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
symmetry c1
""")

psi4.set_options({'basis': 'cc-pvdz'})

# CCSD settings
E_conv = 1.e-8
maxiter = 20
max_diis = 8
compare_psi4 = True
freeze_core = False

# Build CCSD object
ccsd = helper_CCSD(mol, memory=2)

### Setup DIIS
diis_vals_t1 = [ccsd.t1.copy()]
diis_vals_t2 = [ccsd.t2.copy()]
diis_errors = []

### Start Iterations
ccsd_tstart = time.time()

# Compute MP2 energy
CCSDcorr_E_old = ccsd.compute_corr_energy()
print("CCSD Iteration %3d: CCSD correlation = %.12f    dE = % .5E    \
MP2" % (0, CCSDcorr_E_old, -CCSDcorr_E_old))

# Iterate!
diis_size = 0
for CCSD_iter in range(1, maxiter + 1):

    # Save new amplitudes
    oldt1 = ccsd.t1.copy()
    oldt2 = ccsd.t2.copy()

    ccsd.update()

    # Compute CCSD correlation energy
    CCSDcorr_E = ccsd.compute_corr_energy()

    # Print CCSD iteration information
    print('CCSD Iteration %3d: CCSD correlation = %.12f    dE = % .5E    DIIS = %d' % (CCSD_iter, CCSDcorr_E, (CCSDcorr_E - CCSDcorr_E_old), diis_size))

    # Check convergence
    if (abs(CCSDcorr_E - CCSDcorr_E_old) < E_conv):
        break

    # Add DIIS vectors
    diis_vals_t1.append(ccsd.t1.copy())
    diis_vals_t2.append(ccsd.t2.copy())

    # Build new error vector
    error_t1 = (ccsd.t1 - oldt1).ravel()
    error_t2 = (ccsd.t2 - oldt2).ravel()
    diis_errors.append(np.concatenate((error_t1, error_t2)))

    # Update old energy
    CCSDcorr_E_old = CCSDcorr_E

    if CCSD_iter >= 1:

        # Limit size of DIIS vector
        if (len(diis_vals_t1) > max_diis):
            del diis_vals_t1[0]
            del diis_vals_t2[0]
            del diis_errors[0]

        diis_size = len(diis_vals_t1) - 1

        # Build error matrix B, [Pulay:1980:393], Eqn. 6, LHS
        B = np.ones((diis_size + 1, diis_size + 1)) * -1
        B[-1, -1] = 0

        for n1, e1 in enumerate(diis_errors):
            for n2, e2 in enumerate(diis_errors):
                # Vectordot the error vectors
                B[n1, n2] = np.dot(e1, e2)

        B[:-1, :-1] /= np.abs(B[:-1, :-1]).max()

        # Build residual vector, [Pulay:1980:393], Eqn. 6, RHS
        resid = np.zeros(diis_size + 1)
        resid[-1] = -1

        # Solve Pulay equations, [Pulay:1980:393], Eqn. 6
        ci = np.linalg.solve(B, resid)

        # Calculate new amplitudes
        ccsd.t1[:] = 0
        ccsd.t2[:] = 0
        for num in range(diis_size):
            ccsd.t1 += ci[num] * diis_vals_t1[num + 1]
            ccsd.t2 += ci[num] * diis_vals_t2[num + 1]
    # End DIIS amplitude update

# Finished CCSD iterations
print('CCSD iterations took %.2f seconds.\n' % (time.time() - ccsd_tstart))

CCSD_E = ccsd.rhf_e + CCSDcorr_E

print('\nFinal CCSD correlation energy:     % 16.10f' % CCSDcorr_E)
print('Total CCSD energy:                 % 16.10f' % CCSD_E)

if compare_psi4:
    psi4.compare_values(psi4.energy('CCSD'), CCSD_E, 6, 'CCSD Energy')

