# A simple Psi 4 input script to compute SAPT interaction energies
#
# Created by: Daniel G. A. Smith
# Date: 12/1/14
# License: GPL v3.0
#

import time
import numpy as np
from helper_SAPT import *
np.set_printoptions(precision=5, linewidth=200, threshold=2000, suppress=True)
import psi4

# Set Psi4 & NumPy memory options
psi4.core.set_memory(int(2e9), False)
psi4.core.set_output_file('output.dat', False)

numpy_memory = 2

# Set molecule to dimer
dimer = psi4.geometry("""
O   -0.066999140   0.000000000   1.494354740
H    0.815734270   0.000000000   1.865866390
H    0.068855100   0.000000000   0.539142770
--
O    0.062547750   0.000000000  -1.422632080
H   -0.406965400  -0.760178410  -1.771744500
H   -0.406965400   0.760178410  -1.771744500
symmetry c1
""")

psi4.set_options({'basis':'jun-cc-pvdz',
                  'e_convergence':1e-8,
                  'd_convergence':1e-8})

#energy('SAPT2+3')
#raise Exception();
# Knobs
leg_points = 8

sapt = helper_SAPT(psi4, energy, dimer, memory=20)

#### Start E200 Disp
disp_timer = sapt_timer('dispersion')
v_abrs = sapt.v('abrs')
v_rsab = sapt.v('rsab')
e_rsab = 1/(-sapt.eps('r', dim=4) - sapt.eps('s', dim=3) + sapt.eps('a', dim=2) + sapt.eps('b'))

Disp200 = 4 * np.einsum('rsab,rsab,abrs->', e_rsab, v_rsab, v_abrs)
disp_timer.stop()
sapt_printer('Disp200', Disp200)
#### End E200 Disp


def FDDS_coef(omega, v_iajb, v_ijab, eps_o, eps_v):
    ndocc = len(eps_o)
    nvirt = len(eps_v)
    # Hessian
    H1  = np.einsum('ab,ij->iajb', np.diag(eps_v), np.diag(np.ones(ndocc)))
    H1 -= np.einsum('ij,ab->iajb', np.diag(eps_o), np.diag(np.ones(nvirt)))
    H1 += 4 * v_iajb
    H1 -= v_ijab.swapaxes(1, 2)
    H1 -= v_iajb.swapaxes(0, 2)
    
    # H2 
    H2  = np.einsum('ab,ij->iajb', np.diag(eps_v), np.diag(np.ones(ndocc)))
    H2 -= np.einsum('ij,ab->iajb', np.diag(eps_o), np.diag(np.ones(nvirt)))
    H2 += v_ijab.swapaxes(1, 2)
    H2 -= v_iajb.swapaxes(0, 2)

    ov_shape = H1.shape[0] * H1.shape[1]

    inv = (H1 * H2).reshape(ov_shape, ov_shape)
    # Subtract omega^2, since omega is imaginary add omega^2
    inv[np.diag_indices_from(inv)] += omega ** 2

    inv = np.linalg.inv(inv)

    # return -H1 * inv
    H1 *= inv.reshape(H1.shape)
    H1 *= -1
    return H1


total = 0.0

# Build intermediates
v_abrs = sapt.v('abrs', phys=False)

a_v_iajb = sapt.v('arar', phys=False)
a_v_ijab = sapt.v('aarr', phys=False)
a_eps_o = sapt.eps('a')
a_eps_v = sapt.eps('r')

b_v_iajb = sapt.v('bsbs', phys=False)
b_v_ijab = sapt.v('bbss', phys=False)
b_eps_o = sapt.eps('b')
b_eps_v = sapt.eps('s')

disp_timer = sapt_timer('FDDS dispersion')
fdds_lambda = 0.2
print('     Omega      value     weight        sum')
for point, weight in zip(*np.polynomial.legendre.leggauss(leg_points)):
    omega = fdds_lambda * (1.0 - point) / (1.0 + point)

    C_arar = FDDS_coef(omega, a_v_iajb, a_v_ijab, a_eps_o, a_eps_v)
    C_bsbs = FDDS_coef(omega, b_v_iajb, b_v_ijab, b_eps_o, b_eps_v)

    #tmp = np.einsum('abrs,ABRS,arAR,bsBS', v_abrs, v_abrs, C_arar, C_bsbs)
    bsAR = np.einsum('abrs,arAR->bsAR', v_abrs, C_arar)
    BSAR = np.einsum('bsAR,bsBS->BSAR', bsAR, C_bsbs)
    tmp = np.einsum('BSAR,ABRS->', BSAR, v_abrs)

    total += tmp * weight * ( (2 * fdds_lambda) / (point + 1)**2 )
    print('% .3e % .3e % .3e % .3e' % (omega, tmp, weight, tmp*weight))
disp_timer.stop()

Disp2 = total / (-2.0 * np.pi)
sapt_printer('FDDS Disp2', Disp2)


