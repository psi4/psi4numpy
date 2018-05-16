"""
A simple Psi4 input script to compute SAPT0 interaction energies using an
atomic orbital (AO) algorithm.

References:
- Equations and algorithms from [Szalewicz:2005:43], [Jeziorski:1994:1887],
[Szalewicz:2012:254], and [Hohenstein:2012:304]
"""

__authors__   = "Daniel G. A. Smith"
__credits__   = ["Daniel G. A. Smith"]

__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__   = "BSD-3-Clause"
__date__      = "2014-12-01"

import time
import numpy as np
from helper_SAPT import *
np.set_printoptions(precision=5, linewidth=200, threshold=2000, suppress=True)
import psi4

# Set Psi4 & NumPy Memory Options
psi4.set_memory('2 GB')
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

psi4.set_options({'basis': 'jun-cc-pVDZ',
                  'e_convergence': 1e-8,
                  'd_convergence': 1e-8})

sapt = helper_SAPT(dimer, memory=8, algorithm='AO')

# Build intermediates
int_timer = sapt_timer('intermediates')
Pi = np.dot(sapt.orbitals['a'], sapt.orbitals['a'].T)
Pj = np.dot(sapt.orbitals['b'], sapt.orbitals['b'].T)

S = sapt.S
num_el_A = (2 * sapt.ndocc_A)
num_el_B = (2 * sapt.ndocc_B)

Ci = sapt.orbitals['a']
Cj = sapt.orbitals['b']
Cr = sapt.orbitals['r']
Cs = sapt.orbitals['s']

I = np.asarray(sapt.mints.ao_eri())

Jii, Kii = sapt.compute_sapt_JK(Ci, Ci)
Jjj, Kjj = sapt.compute_sapt_JK(Cj, Cj)

Jij, Kij = sapt.compute_sapt_JK(Ci, Cj, tensor=sapt.chain_dot(Ci.T, S, Cj))

w_A = sapt.V_A + 2 * Jii 
w_B = sapt.V_B + 2 * Jjj 

h_A = sapt.V_A + 2 * Jii - Kii 
h_B = sapt.V_B + 2 * Jjj - Kjj 

int_timer.stop()

### Build electrostatics
elst_timer = sapt_timer('electrostatics')
two_el = 2 * np.vdot(Pi, Jjj)
att_a = np.vdot(sapt.V_A, Pj) 
att_b = np.vdot(sapt.V_B, Pi) 
rep = sapt.nuc_rep 
elst_ijij = 2 * (two_el + att_a + att_b) + rep

Elst10 = elst_ijij
sapt_printer('Elst10', Elst10)
elst_timer.stop()
### End electrostatics

### Start exchange
exch_timer = sapt_timer('exchange')
exch = 0
exch -= 2 * np.vdot(Pj, Kii)
exch -= 2 * np.vdot(sapt.chain_dot(Pi, S, Pj), (h_A + h_B))

exch += 2 * np.vdot(sapt.chain_dot(Pj, S, Pi, S, Pj), w_A)
exch += 2 * np.vdot(sapt.chain_dot(Pi, S, Pj, S, Pi), w_B)

exch -= 2 * np.vdot(sapt.chain_dot(Pi, S, Pj), Kij)

Exch100 = exch
sapt_printer('Exch10(S^2)', Exch100)
exch_timer.stop()
### End E100 (S^2) Exchange

### Start E200 Disp
disp_timer = sapt_timer('dispersion')
v_abrs = sapt.v('abrs')
v_rsab = sapt.v('rsab')
e_rsab = 1/(-sapt.eps('r', dim=4) - sapt.eps('s', dim=3) + sapt.eps('a', dim=2) + sapt.eps('b'))

Disp200 = 4 * np.einsum('rsab,rsab,abrs->', e_rsab, v_rsab, v_abrs)
sapt_printer('Disp20', Disp200)
### End E200 Disp

### Start E200 Exchange-Dispersion

# Build t_rsab
t_rsab = np.einsum('rsab,rsab->rsab', v_rsab, e_rsab)

#backtransform t_rsab to the AO basis
t_lsab = np.einsum('rsab,rl->lsab', t_rsab, Cr.T)
t_lnab = np.einsum('lsab,sn->lnab', t_lsab, Cs.T)
t_lnkb = np.einsum('lnab,ak->lnkb', t_lnab, Ci.T)
t_lnkm = np.einsum('lnkb,bm->lnkm', t_lnkb, Cj.T)

ExchDisp20 = - 2 * np.einsum('lnkm,knml->', t_lnkm, I)

ExchDisp20 -= 2 * np.einsum('lnkm,ml,kn->', t_lnkm, h_A, S) 
ExchDisp20 -= 2 * np.einsum('lnkm,ml,kn->', t_lnkm, S, h_B)

interm = 2 * np.einsum('klmq,nq->klmn', I, np.dot(S, Pi)) 
ExchDisp20 -= 2 * np.einsum('lnkm,klmn->', t_lnkm, interm) 
ExchDisp20 += np.einsum('lnkm,mlkn->', t_lnkm, interm) 

interm = 2 * np.einsum('klmq,nq->klmn', I, np.dot(S, Pj)) 
ExchDisp20 -= 2 * np.einsum('lnkm,mnkl->', t_lnkm, interm) 
ExchDisp20 += np.einsum('lnkm,knml->', t_lnkm, interm) 

ExchDisp20 -= 4 * np.einsum('lnkm,mn,kl->', t_lnkm, w_A, sapt.chain_dot(S, Pj, S))
ExchDisp20 += 2 * np.einsum('lnkm,kn,ml->', t_lnkm, S, sapt.chain_dot(w_A, Pj, S))
ExchDisp20 += 2 * np.einsum('lnkm,ml,nk->', t_lnkm, S, sapt.chain_dot(w_A, Pj, S))

ExchDisp20 -= 4 * np.einsum('lnkm,kl,mn->', t_lnkm, w_B, sapt.chain_dot(S, Pi, S))
ExchDisp20 += 2 * np.einsum('lnkm,ml,kn->', t_lnkm, S, sapt.chain_dot(w_B, Pi, S))
ExchDisp20 += 2 * np.einsum('lnkm,kn,lm->', t_lnkm, S, sapt.chain_dot(w_B, Pi, S))

spbspa = sapt.chain_dot(S, Pj, S, Pi)
spaspb = sapt.chain_dot(S, Pi, S, Pj)
interm = np.einsum('kqmn,lq->klmn', I, spbspa)
interm += np.einsum('plmn,kp->klmn', I, spbspa)
interm += np.einsum('klms,ns->klmn', I, spaspb)
interm += np.einsum('klrn,mr->klmn', I, spaspb)
ExchDisp20 += 4 * np.einsum('lnkm,klmn->', t_lnkm, interm)

ExchDisp20 -= 2 * np.einsum('lnkm,kn,ml->', t_lnkm, S, Kij.T) 
ExchDisp20 -= 2 * np.einsum('lnkm,ml,nk->', t_lnkm, S, Kij.T) 

spa = np.dot(S, Pi)
spb = np.dot(S, Pj)
interm = np.einsum('kpmq,nq->kpmn', I, spa)
interm = np.einsum('kpmn,lp->klmn', interm, spb)
ExchDisp20 -= 2 * np.einsum('lnkm,mlkn->', t_lnkm, interm)
ExchDisp20 -= 2 * np.einsum('lnkm,nklm->', t_lnkm, interm)

sapt_printer('Exch-Disp20', ExchDisp20)

disp_timer.stop()
### End E200 Exchange-Dispersion


### Start E200 Induction and Exchange-Induction

# E200Induction and CPHF orbitals
ind_timer = sapt_timer('induction')

CPHF_ra, Ind20_ba = sapt.chf('B', ind=True)
sapt_printer('Ind20,r (A<-B)', Ind20_ba)

CPHF_sb, Ind20_ab = sapt.chf('A', ind=True)
sapt_printer('Ind20,r (A->B)', Ind20_ab)

Ind20r = Ind20_ba + Ind20_ab

# Exchange-Induction

# A <- B
CPHFA_ao = sapt.chain_dot(Ci, CPHF_ra.T, Cr.T)
ExchInd20_ab = -2 * np.vdot(CPHFA_ao, Kjj)
ExchInd20_ab -= 2 * np.vdot(CPHFA_ao, sapt.chain_dot(S, Pj, h_A))
ExchInd20_ab -= 2 * np.vdot(CPHFA_ao, sapt.chain_dot(h_B, Pj, S))

ExchInd20_ab -= 4 * np.vdot(CPHFA_ao, Jij)
ExchInd20_ab += 2 * np.vdot(CPHFA_ao, Kij)

ExchInd20_ab += 2 * np.vdot(CPHFA_ao, sapt.chain_dot(w_B, Pi, S, Pj, S))
ExchInd20_ab += 2 * np.vdot(CPHFA_ao, sapt.chain_dot(S, Pj, S, Pi, w_B))
ExchInd20_ab += 2 * np.vdot(CPHFA_ao, sapt.chain_dot(S, Pj, w_A, Pj, S))

Jjij, Kjij = sapt.compute_sapt_JK(Cj, Cj, tensor=sapt.chain_dot(Cj.T, S, Pi, S, Cj))

ExchInd20_ab += 4 * np.vdot(CPHFA_ao, Jjij)
ExchInd20_ab -= 2 * np.vdot(CPHFA_ao, sapt.chain_dot(S, Pj, Kij.T))
ExchInd20_ab -= 2 * np.vdot(CPHFA_ao, sapt.chain_dot(Kij, Pj, S))

sapt_printer('Exch-Ind20,r (A<-B)', ExchInd20_ab)

# B <- A
CPHFB_ao = sapt.chain_dot(Cj, CPHF_sb.T, Cs.T)
ExchInd20_ba = -2 * np.vdot(CPHFB_ao, Kii)
ExchInd20_ba -= 2 * np.vdot(CPHFB_ao, sapt.chain_dot(S, Pi, h_B))
ExchInd20_ba -= 2 * np.vdot(CPHFB_ao, sapt.chain_dot(h_A, Pi, S))

ExchInd20_ba -= 4 * np.vdot(CPHFB_ao, Jij)
ExchInd20_ba += 2 * np.vdot(CPHFB_ao, Kij.T)

ExchInd20_ba += 2 * np.vdot(CPHFB_ao, sapt.chain_dot(w_A, Pj, S, Pi, S))
ExchInd20_ba += 2 * np.vdot(CPHFB_ao, sapt.chain_dot(S, Pi, S, Pj, w_A))
ExchInd20_ba += 2 * np.vdot(CPHFB_ao, sapt.chain_dot(S, Pi, w_B, Pi, S))

Jiji, Kiji = sapt.compute_sapt_JK(Ci, Ci, tensor=sapt.chain_dot(Ci.T, S, Pj, S, Ci))

ExchInd20_ba += 4 * np.vdot(CPHFB_ao, Jiji)
ExchInd20_ba -= 2 * np.vdot(CPHFB_ao, sapt.chain_dot(S, Pi, Kij))
ExchInd20_ba -= 2 * np.vdot(CPHFB_ao, sapt.chain_dot(Kij.T, Pi, S))

sapt_printer('Exch-Ind20,r (A->B)', ExchInd20_ba)
ExchInd20r = ExchInd20_ba + ExchInd20_ab

ind_timer.stop()
### End E200 Induction and Exchange-Induction

print('\nSAPT0 Results')
print('-' * 70)
sapt_printer('Exch10 (S^2)', Exch100)
sapt_printer('Elst10', Elst10)
sapt_printer('Disp20', Disp200)
sapt_printer('Exch-Disp20', ExchDisp20)
sapt_printer('Ind20,r', Ind20r)
sapt_printer('Exch-Ind20,r', ExchInd20r)

print('-' * 70)
sapt0 = Exch100 + Elst10 + Disp200 + ExchDisp20 + Ind20r + ExchInd20r
sapt_printer('Total SAPT0', sapt0)
