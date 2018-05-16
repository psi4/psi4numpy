"""
A simple Psi4 script to compute SAPT0 interaction energies.

References:
- Equations and algorithms from [Szalewicz:2005:43], [Jeziorski:1994:1887],
[Szalewicz:2012:254], and [Hohenstein:2012:304]
"""

__authors__   =  "Daniel G. A. Smith"
__credits__   =  ["Daniel G. A. Smith"]

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

sapt = helper_SAPT(dimer, memory=8)

### Start E100 Electostatics
elst_timer = sapt_timer('electrostatics')
Elst10 = 4 * np.einsum('abab', sapt.vt('abab'))
elst_timer.stop()
### End E100 Electrostatics

### Start E100 Exchange
exch_timer = sapt_timer('exchange')
vt_abba = sapt.vt('abba')
vt_abaa = sapt.vt('abaa')
vt_abbb = sapt.vt('abbb')
vt_abab = sapt.vt('abab')
s_ab = sapt.s('ab')

Exch100 = np.einsum('abba', vt_abba)

tmp = 2 * vt_abaa - vt_abaa.swapaxes(2, 3)
Exch100 += np.einsum('Ab,abaA', s_ab, tmp)

tmp = 2 * vt_abbb - vt_abbb.swapaxes(2, 3)
Exch100 += np.einsum('Ba,abBb', s_ab.T, tmp)

Exch100 -= 2 * np.einsum('Ab,BA,abaB', s_ab, s_ab.T, vt_abab)
Exch100 -= 2 * np.einsum('AB,Ba,abAb', s_ab, s_ab.T, vt_abab)
Exch100 += np.einsum('Ab,Ba,abAB', s_ab, s_ab.T, vt_abab)

Exch100 *= -2
exch_timer.stop()
### End E100 (S^2) Exchange

### Start E200 Disp
disp_timer = sapt_timer('dispersion')
v_abrs = sapt.v('abrs')
v_rsab = sapt.v('rsab')
e_rsab = 1/(-sapt.eps('r', dim=4) - sapt.eps('s', dim=3) + sapt.eps('a', dim=2) + sapt.eps('b'))

Disp200 = 4 * np.einsum('rsab,rsab,abrs->', e_rsab, v_rsab, v_abrs)
### End E200 Disp

### Start E200 Exchange-Dispersion

# Build t_rsab
t_rsab = np.einsum('rsab,rsab->rsab', v_rsab, e_rsab)

# Build h_abrs
vt_abar = sapt.vt('abar')
vt_abra = sapt.vt('abra')
vt_absb = sapt.vt('absb')
vt_abbs = sapt.vt('abbs')

tmp = 2 * vt_abar - vt_abra.swapaxes(2, 3)
h_abrs = np.einsum('as,AbAr->abrs', sapt.s('as'), tmp)

tmp = 2 * vt_abra - vt_abar.swapaxes(2, 3)
h_abrs += np.einsum('As,abrA->abrs', sapt.s('as'), tmp)

tmp = 2 * vt_absb - vt_abbs.swapaxes(2, 3)
h_abrs += np.einsum('br,aBsB->abrs', sapt.s('br'), tmp)

tmp = 2 * vt_abbs - vt_absb.swapaxes(2, 3)
h_abrs += np.einsum('Br,abBs->abrs', sapt.s('br'), tmp)

# Build q_abrs
vt_abas = sapt.vt('abas')
q_abrs =      np.einsum('br,AB,aBAs->abrs', sapt.s('br'), sapt.s('ab'), vt_abas)
q_abrs -= 2 * np.einsum('Br,AB,abAs->abrs', sapt.s('br'), sapt.s('ab'), vt_abas)
q_abrs -= 2 * np.einsum('br,aB,ABAs->abrs', sapt.s('br'), sapt.s('ab'), vt_abas)
q_abrs += 4 * np.einsum('Br,aB,AbAs->abrs', sapt.s('br'), sapt.s('ab'), vt_abas)

vt_abrb = sapt.vt('abrb')
q_abrs -= 2 * np.einsum('as,bA,ABrB->abrs', sapt.s('as'), sapt.s('ba'), vt_abrb)
q_abrs += 4 * np.einsum('As,bA,aBrB->abrs', sapt.s('as'), sapt.s('ba'), vt_abrb)
q_abrs +=     np.einsum('as,BA,AbrB->abrs', sapt.s('as'), sapt.s('ba'), vt_abrb)
q_abrs -= 2 * np.einsum('As,BA,abrB->abrs', sapt.s('as'), sapt.s('ba'), vt_abrb)

vt_abab = sapt.vt('abab')
q_abrs +=     np.einsum('Br,As,abAB->abrs', sapt.s('br'), sapt.s('as'), vt_abab)
q_abrs -= 2 * np.einsum('br,As,aBAB->abrs', sapt.s('br'), sapt.s('as'), vt_abab)
q_abrs -= 2 * np.einsum('Br,as,AbAB->abrs', sapt.s('br'), sapt.s('as'), vt_abab)

vt_abrs = sapt.vt('abrs')
q_abrs +=     np.einsum('bA,aB,ABrs->abrs', sapt.s('ba'), sapt.s('ab'), vt_abrs)
q_abrs -= 2 * np.einsum('bA,AB,aBrs->abrs', sapt.s('ba'), sapt.s('ab'), vt_abrs)
q_abrs -= 2 * np.einsum('BA,aB,Abrs->abrs', sapt.s('ba'), sapt.s('ab'), vt_abrs)

# Sum it all together
xd_absr = sapt.vt('absr')
xd_absr += h_abrs.swapaxes(2, 3)
xd_absr += q_abrs.swapaxes(2, 3)
ExchDisp20 = -2 * np.einsum('absr,rsab->', xd_absr, t_rsab)

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
vt_abra = sapt.vt('abra')
vt_abar = sapt.vt('abar')
ExchInd20_ab  =     np.einsum('ra,abbr', CPHF_ra, sapt.vt('abbr'))
ExchInd20_ab += 2 * np.einsum('rA,Ab,abar', CPHF_ra, sapt.s('ab'), vt_abar)
ExchInd20_ab += 2 * np.einsum('ra,Ab,abrA', CPHF_ra, sapt.s('ab'), vt_abra)
ExchInd20_ab -=     np.einsum('rA,Ab,abra', CPHF_ra, sapt.s('ab'), vt_abra)

vt_abbb = sapt.vt('abbb')
vt_abab = sapt.vt('abab')
ExchInd20_ab -=     np.einsum('ra,Ab,abAr', CPHF_ra, sapt.s('ab'), vt_abar)
ExchInd20_ab += 2 * np.einsum('ra,Br,abBb', CPHF_ra, sapt.s('br'), vt_abbb)
ExchInd20_ab -=     np.einsum('ra,Br,abbB', CPHF_ra, sapt.s('br'), vt_abbb)
ExchInd20_ab -= 2 * np.einsum('rA,Ab,Br,abaB', CPHF_ra, sapt.s('ab'), sapt.s('br'), vt_abab)

vt_abrb = sapt.vt('abrb')
ExchInd20_ab -= 2 * np.einsum('ra,Ab,BA,abrB', CPHF_ra, sapt.s('ab'), sapt.s('ba'), vt_abrb)
ExchInd20_ab -= 2 * np.einsum('ra,AB,Br,abAb', CPHF_ra, sapt.s('ab'), sapt.s('br'), vt_abab)
ExchInd20_ab -= 2 * np.einsum('rA,AB,Ba,abrb', CPHF_ra, sapt.s('ab'), sapt.s('ba'), vt_abrb)

ExchInd20_ab +=     np.einsum('ra,Ab,Br,abAB', CPHF_ra, sapt.s('ab'), sapt.s('br'), vt_abab)
ExchInd20_ab +=     np.einsum('rA,Ab,Ba,abrB', CPHF_ra, sapt.s('ab'), sapt.s('ba'), vt_abrb)

ExchInd20_ab *= -2
sapt_printer('Exch-Ind20,r (A<-B)', ExchInd20_ab)

# B <- A
vt_abbs = sapt.vt('abbs')
vt_absb = sapt.vt('absb')
ExchInd20_ba  =     np.einsum('sb,absa', CPHF_sb, sapt.vt('absa'))
ExchInd20_ba += 2 * np.einsum('sB,Ba,absb', CPHF_sb, sapt.s('ba'), vt_absb)
ExchInd20_ba += 2 * np.einsum('sb,Ba,abBs', CPHF_sb, sapt.s('ba'), vt_abbs)
ExchInd20_ba -=     np.einsum('sB,Ba,abbs', CPHF_sb, sapt.s('ba'), vt_abbs)

vt_abaa = sapt.vt('abaa')
vt_abab = sapt.vt('abab')
ExchInd20_ba -=     np.einsum('sb,Ba,absB', CPHF_sb, sapt.s('ba'), vt_absb)
ExchInd20_ba += 2 * np.einsum('sb,As,abaA', CPHF_sb, sapt.s('as'), vt_abaa)
ExchInd20_ba -=     np.einsum('sb,As,abAa', CPHF_sb, sapt.s('as'), vt_abaa)
ExchInd20_ba -= 2 * np.einsum('sB,Ba,As,abAb', CPHF_sb, sapt.s('ba'), sapt.s('as'), vt_abab)

vt_abas = sapt.vt('abas')
ExchInd20_ba -= 2 * np.einsum('sb,Ba,AB,abAs', CPHF_sb, sapt.s('ba'), sapt.s('ab'), vt_abas)
ExchInd20_ba -= 2 * np.einsum('sb,BA,As,abaB', CPHF_sb, sapt.s('ba'), sapt.s('as'), vt_abab)
ExchInd20_ba -= 2 * np.einsum('sB,BA,Ab,abas', CPHF_sb, sapt.s('ba'), sapt.s('ab'), vt_abas)

ExchInd20_ba +=     np.einsum('sb,Ba,As,abAB', CPHF_sb, sapt.s('ba'), sapt.s('as'), vt_abab)
ExchInd20_ba +=     np.einsum('sB,Ba,Ab,abAs', CPHF_sb, sapt.s('ba'), sapt.s('ab'), vt_abas)

ExchInd20_ba *= -2
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
