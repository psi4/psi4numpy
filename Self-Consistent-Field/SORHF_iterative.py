# A simple Psi 4 input script to compute a SCF reference using second-order optimizations
# Requires scipy numpy 1.7.2+
#
# Created by: Daniel G. A. Smith
# Date: 2/27/15
# License: GPL v3.0
#

import time
import numpy as np
np.set_printoptions(precision=3, linewidth=200, suppress=True)
from helper_HF import *
import psi4

# Memory for Psi4 in GB
psi4.set_memory('2 GB')
psi4.core.set_output_file('output.dat', False)

# Memory for numpy in GB
numpy_memory = 2

mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 109
symmetry c1
""")

psi4.set_options({'scf_type': 'df',
                  'basis': 'aug-cc-pvdz',
                  'e_convergence': 1e1,
                  'd_convergence': 1e1})

# Knobs
E_conv = 1.e-8
D_conv = 1.e-4
max_macro = 10
max_micro = 3
micro_conv = 1.e-3
micro_print = True

# Build objects
diis = DIIS_helper()
hf = helper_HF(mol, scf_type='DF', guess='CORE')
ndocc = hf.ndocc
nvirt = hf.nvirt

print('\nStart SCF iterations:\n')
t = time.time()
E = 0.0
Eold = 0.0
iter_type = 'CORE'

def SCF_Hx(x, moF, Co, Cv):
    """
    Compute a hessian vector guess where x is a ov matrix of nonredundant operators.
    """
    F  = np.dot(moF[:ndocc, :ndocc], x)
    F -= np.dot(x, moF[ndocc:, ndocc:])

    # Build two electron part, M = -4 (4 G_{mnip} - g_{mpin} - g_{npim}) K_{ip}
    C_right = np.einsum('ia,sa->si', -x, Cv)
    J, K = hf.build_jk(Co, C_right)
    F  += (Co.T).dot(4 * J - K.T - K).dot(Cv)
    F *= -4
    return F

for SCF_ITER in range(1, max_macro):

    # Build new fock matrix
    F = hf.build_fock()

    # DIIS error and update
    diis_e = F.dot(hf.Da).dot(hf.S) - hf.S.dot(hf.Da).dot(F)
    diis_e = (hf.A).dot(diis_e).dot(hf.A)
    diis.add(F, diis_e)

    # SCF energy and update
    scf_e = hf.compute_hf_energy()
    dRMS = np.mean(diis_e ** 2) ** 0.5
    print('SCF Iteration %3d: Energy = %4.16f   dE = % 1.5E   dRMS = %1.5E   %s' % (SCF_ITER, hf.scf_e, (hf.scf_e - Eold), dRMS, iter_type))
    if (abs(hf.scf_e - Eold) < E_conv) and (dRMS < D_conv):
        break

    Eold = hf.scf_e

    # Build MO fock ,matrix and gradient
    Co = hf.Ca[:, :ndocc]
    Cv = hf.Ca[:, ndocc:]
    moF = np.einsum('ui,vj,uv->ij', hf.Ca, hf.Ca, F)
    gradient = -4 * moF[:ndocc, ndocc:]
    grad_dot = np.vdot(gradient, gradient)

    if (np.max(np.abs(gradient)) > 0.2):
        F = diis.extrapolate()
        eps, C = hf.diag(F)
        hf.set_Cleft(C)
        iter_type = 'DIIS'
    else:

        # Initial guess
        eps = np.diag(moF)
        precon = -4 * (eps[:ndocc].reshape(-1, 1) - eps[ndocc:])

        x = gradient / precon
        Ax = SCF_Hx(x, moF, Co, Cv)
        r = gradient - Ax
        z = r / precon
        p = z.copy()
        rms = (np.vdot(r, r) / grad_dot) ** 0.5
        if micro_print:
            print('Micro Iteration Guess: Rel. RMS = %1.5e' %  (rms))

        # CG iterations
        for rot_iter in range(max_micro):
            rz_old = np.vdot(r, z)

            Ap = SCF_Hx(p, moF, Co, Cv)
            alpha = rz_old / np.vdot(Ap, p)

            x += alpha * p
            r -= alpha * Ap
            z = r / precon

            rms = (np.vdot(r, r) / grad_dot) ** 0.5

            if micro_print:
                print('Micro Iteration %5d: Rel. RMS = %1.5e' %  (rot_iter + 1, rms))
            if rms < micro_conv:
                break

            beta = np.vdot(r, z) / rz_old
            p = z + beta * p

        C = rotate_orbitals(hf.Ca, x)

        hf.set_Cleft(C)
        iter_type = 'SOSCF, nmicro ' + str(rot_iter + 1)

print('Total time taken for SCF iterations: %.3f seconds \n' % (time.time()-t))

print('Final SCF energy:     %.8f hartree' % hf.scf_e)

# Compute w/ Psi4 and compare
psi4.set_options({'e_convergence': 1e-7,
                  'd_convergence': 1e-7})

SCF_E_psi = psi4.energy('SCF')
psi4.driver.p4util.compare_values(SCF_E_psi, hf.scf_e, 6, 'SCF Energy')


