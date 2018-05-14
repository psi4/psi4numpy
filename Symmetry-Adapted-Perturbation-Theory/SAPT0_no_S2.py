'''
A script to compute the SAPT0 interaction energy without the
Single-Exchange Approximation.

References
Equations taken from [Jeziorski:1976:281], [Schaffer:2012:1235],
and [Schaffer:2013:2570].
'''

__authors__ = "Jonathan M. Waldrop"
__credits__ = ["Jonathan M. Waldrop"]

__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__ = "BSD-3-Clause"
__date__ = "2018-01-09"

import psi4
import numpy as np
np.set_printoptions(precision=5, linewidth=200, threshold=2000, suppress=True)
from helper_SAPT import *

# Set Psi4 & NumPy Memory Options
psi4.set_memory('2 GB')
psi4.core.set_output_file('sapt0_no_S2.dat', False)

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

psi4.set_options({'basis': 'jun-cc-pVDZ', 'e_convergence': 1e-8, 'd_convergence': 1e-8})

sapt = helper_SAPT(dimer, memory=8)

### Overlap Matrix and Inverse
S_a = np.concatenate((sapt.s('aa'), sapt.s('ab')), axis=1)
S_b = np.concatenate((sapt.s('ba'), sapt.s('bb')), axis=1)
S = np.concatenate((S_a, S_b), axis=0)
# S_{AA} S_{AB}
# S_{BA} S_{BB}

D = np.linalg.inv(S)
D_aa = D[:sapt.ndocc_A, :sapt.ndocc_A]
D_ab = D[:sapt.ndocc_A, sapt.ndocc_A:sapt.ndocc_A + sapt.ndocc_B]
D_ba = D[sapt.ndocc_A:sapt.ndocc_A + sapt.ndocc_B, :sapt.ndocc_A]
D_bb = D[sapt.ndocc_A:, sapt.ndocc_A:]

### E10 Electrostatics
Elst10 = 4 * np.einsum('abab', sapt.vt('abab'))

### Complete E10
v_abaa = sapt.v('abaa')
v_abab = sapt.v('abab')
v_abba = sapt.v('abba')
v_abbb = sapt.v('abbb')

# E10 Full
e1_full = sapt.nuc_rep  # Nuclear Repulsion

e1_full += 2 * (np.einsum('aA,Aa->', sapt.potential('aa', 'B'), D_aa) +
                np.einsum('ab,ba->', sapt.potential('ab', 'B'), D_ba))  # B potential
e1_full += 2 * (np.einsum('bB,Bb->', sapt.potential('bb', 'A'), D_bb) +
                np.einsum('ba,ab->', sapt.potential('ba', 'A'), D_ab))  # A potential

e1_full += 4 * np.einsum('ijkl,ki,lj->', v_abaa, D_aa, D_ab)  # Two electron part
e1_full += 4 * np.einsum('ijkl,ki,lj->', v_abab, D_aa, D_bb)
e1_full += 4 * np.einsum('ijkl,ki,lj->', v_abba, D_ba, D_ab)
e1_full += 4 * np.einsum('ijkl,ki,lj->', v_abbb, D_ba, D_bb)
e1_full += -2 * np.einsum('ijlk,ki,lj->', v_abaa, D_aa, D_ab)
e1_full += -2 * np.einsum('ijlk,ki,lj->', v_abba, D_aa, D_bb)
e1_full += -2 * np.einsum('ijlk,ki,lj->', v_abab, D_ba, D_ab)
e1_full += -2 * np.einsum('ijlk,ki,lj->', v_abbb, D_ba, D_bb)

# E10 Exchange
Exch10 = e1_full - Elst10

### E20 Induction and Exchange Induction

# E20 Induction
CPHF_ra, Ind20_ba = sapt.chf('B', ind=True)
CPHF_sb, Ind20_ab = sapt.chf('A', ind=True)
Ind20r = Ind20_ba + Ind20_ab

# E20 Induction Full
T_ar = np.einsum('ij,jk->ik', D_ab, sapt.s('br'))
T_br = np.einsum('ij,jk->ik', D_bb, sapt.s('br'))
T_as = np.einsum('ij,jk->ik', D_aa, sapt.s('as'))
T_bs = np.einsum('ij,jk->ik', D_ba, sapt.s('as'))

B_aa = sapt.potential('aa', 'B')
B_ab = sapt.potential('ab', 'B')
A_ba = sapt.potential('ba', 'A')
A_bb = sapt.potential('bb', 'A')

B_T_ar = np.einsum('ij,jk->ik', B_aa, T_ar) + np.einsum('ij,jk->ik', B_ab, T_br)
B_T_as = np.einsum('ij,jk->ik', B_aa, T_as) + np.einsum('ij,jk->ik', B_ab, T_bs)
A_T_br = np.einsum('ij,jk->ik', A_ba, T_ar) + np.einsum('ij,jk->ik', A_bb, T_br)
A_T_bs = np.einsum('ij,jk->ik', A_ba, T_as) + np.einsum('ij,jk->ik', A_bb, T_bs)

Bt_ar = sapt.potential('ar', 'B') - B_T_ar
Bt_as = sapt.potential('as', 'B') - B_T_as
At_br = sapt.potential('br', 'A') - A_T_br
At_bs = sapt.potential('bs', 'A') - A_T_bs

v_abaa = sapt.v('abaa')
v_abab = sapt.v('abab')
v_abba = sapt.v('abba')
v_abbb = sapt.v('abbb')

v_abra = sapt.v('abra')
v_abar = sapt.v('abar')
v_abrb = sapt.v('abrb')
v_abbr = sapt.v('abbr')

v_absa = sapt.v('absa')
v_abas = sapt.v('abas')
v_absb = sapt.v('absb')
v_abbs = sapt.v('abbs')

v_T_abra = np.einsum('ijkl,ka->ijal', v_abaa, T_ar) + np.einsum('ijkl,ka->ijal', v_abba, T_br)
v_T_abrb = np.einsum('ijkl,ka->ijal', v_abab, T_ar) + np.einsum('ijkl,ka->ijal', v_abbb, T_br)
v_T_abar = np.einsum('ijkl,la->ijka', v_abaa, T_ar) + np.einsum('ijkl,la->ijka', v_abab, T_br)
v_T_abbr = np.einsum('ijkl,la->ijka', v_abba, T_ar) + np.einsum('ijkl,la->ijka', v_abbb, T_br)

v_T_absa = np.einsum('ijkl,ka->ijal', v_abaa, T_as) + np.einsum('ijkl,ka->ijal', v_abba, T_bs)
v_T_absb = np.einsum('ijkl,ka->ijal', v_abab, T_as) + np.einsum('ijkl,ka->ijal', v_abbb, T_bs)
v_T_abas = np.einsum('ijkl,la->ijka', v_abaa, T_as) + np.einsum('ijkl,la->ijka', v_abab, T_bs)
v_T_abbs = np.einsum('ijkl,la->ijka', v_abba, T_as) + np.einsum('ijkl,la->ijka', v_abbb, T_bs)

vt_abra = v_abra - v_T_abra
vt_abar = v_abar - v_T_abar
vt_abrb = v_abrb - v_T_abrb
vt_abbr = v_abbr - v_T_abbr

vt_absa = v_absa - v_T_absa
vt_abas = v_abas - v_T_abas
vt_absb = v_absb - v_T_absb
vt_abbs = v_abbs - v_T_abbs

O_ar = 2 * np.einsum('kj,ik->ij', Bt_ar, D_aa) + 2 * np.einsum('kj,ik->ij', At_br, D_ab)
O_ar += 4 * np.einsum('ijkl,mi,lj->mk', vt_abra, D_aa, D_ab)
O_ar += 4 * np.einsum('ijkl,mi,lj->mk', vt_abrb, D_aa, D_bb)
O_ar -= 2 * np.einsum('ijkl,mj,li->mk', vt_abra, D_ab, D_aa)
O_ar -= 2 * np.einsum('ijkl,mj,li->mk', vt_abrb, D_ab, D_ba)
O_ar -= 2 * np.einsum('ijkl,mi,kj->ml', vt_abar, D_aa, D_ab)
O_ar -= 2 * np.einsum('ijkl,mi,kj->ml', vt_abbr, D_aa, D_bb)
O_ar += 4 * np.einsum('ijkl,mj,ki->ml', vt_abar, D_ab, D_aa)
O_ar += 4 * np.einsum('ijkl,mj,ki->ml', vt_abbr, D_ab, D_ba)

O_bs = 2 * np.einsum('kj,ik->ij', Bt_as, D_ba) + 2 * np.einsum('kj,ik->ij', At_bs, D_bb)
O_bs += 4 * np.einsum('ijkl,mj,ki->ml', vt_abas, D_bb, D_aa)
O_bs += 4 * np.einsum('ijkl,mj,ki->ml', vt_abbs, D_bb, D_ba)
O_bs -= 2 * np.einsum('ijkl,mi,kj->ml', vt_abas, D_ba, D_ab)
O_bs -= 2 * np.einsum('ijkl,mi,kj->ml', vt_abbs, D_ba, D_bb)
O_bs -= 2 * np.einsum('ijkl,mj,li->mk', vt_absa, D_bb, D_aa)
O_bs -= 2 * np.einsum('ijkl,mj,li->mk', vt_absb, D_bb, D_ba)
O_bs += 4 * np.einsum('ijkl,mi,lj->mk', vt_absa, D_ba, D_ab)
O_bs += 4 * np.einsum('ijkl,mi,lj->mk', vt_absb, D_ba, D_bb)

e2_ind_full = np.einsum('ar,ra->', O_ar, CPHF_ra) + np.einsum('bs,sb->', O_bs, CPHF_sb)

# E20 Exchange-Induction
ExchInd20r = e2_ind_full - Ind20r

### E20 Dispersion and Exchange-Dispersion
# E20 Dispersion
v_abrs = sapt.v('abrs')
v_rsab = sapt.v('rsab')
e_rsab = 1 / (-sapt.eps('r', dim=4) - sapt.eps('s', dim=3) + sapt.eps('a', dim=2) + sapt.eps('b'))

Disp20 = 4 * np.einsum('rsab,rsab,abrs->', e_rsab, v_rsab, v_abrs)

# E20 Dispersion Full. Several pieces already produced in E20 Induction Full.
v_T_abRs = np.einsum('ijkl,ka->ijal', v_abas, T_ar) + np.einsum('ijkl,ka->ijal', v_abbs, T_br)
v_T_absR = np.einsum('ijkl,la->ijka', v_absa, T_ar) + np.einsum('ijkl,la->ijka', v_absb, T_br)
v_T_abrS = np.einsum('ijkl,la->ijka', v_abra, T_as) + np.einsum('ijkl,la->ijka', v_abrb, T_bs)
v_T_abSr = np.einsum('ijkl,ka->ijal', v_abar, T_as) + np.einsum('ijkl,ka->ijal', v_abbr, T_bs)

v_T_T_abRS = (np.einsum('ijkl,ka,lb->ijab', v_abaa, T_ar, T_as) + np.einsum('ijkl,ka,lb->ijab', v_abbb, T_br, T_bs) +
              np.einsum('ijkl,ka,lb->ijab', v_abba, T_br, T_as) + np.einsum('ijkl,ka,lb->ijab', v_abab, T_ar, T_bs))
v_T_T_abSR = (np.einsum('ijkl,ka,lb->ijab', v_abaa, T_as, T_ar) + np.einsum('ijkl,ka,lb->ijab', v_abbb, T_bs, T_br) +
              np.einsum('ijkl,ka,lb->ijab', v_abba, T_bs, T_ar) + np.einsum('ijkl,ka,lb->ijab', v_abab, T_as, T_br))

vt_abrs = sapt.v('abrs') - v_T_abRs - v_T_abrS + v_T_T_abRS
vt_absr = sapt.v('absr') - v_T_absR - v_T_abSr + v_T_T_abSR

O_as = 2 * np.einsum('kj,ik->ij', Bt_as, D_aa) + 2 * np.einsum('kj,ik->ij', At_bs, D_ab)
O_as += 4 * np.einsum('ijkl,mi,lj->mk', vt_absa, D_aa, D_ab)
O_as += 4 * np.einsum('ijkl,mi,lj->mk', vt_absb, D_aa, D_bb)
O_as -= 2 * np.einsum('ijkl,mj,li->mk', vt_absa, D_ab, D_aa)
O_as -= 2 * np.einsum('ijkl,mj,li->mk', vt_absb, D_ab, D_ba)
O_as -= 2 * np.einsum('ijkl,mi,kj->ml', vt_abas, D_aa, D_ab)
O_as -= 2 * np.einsum('ijkl,mi,kj->ml', vt_abbs, D_aa, D_bb)
O_as += 4 * np.einsum('ijkl,mj,ki->ml', vt_abas, D_ab, D_aa)
O_as += 4 * np.einsum('ijkl,mj,ki->ml', vt_abbs, D_ab, D_ba)

O_br = 2 * np.einsum('kj,ik->ij', Bt_ar, D_ba) + 2 * np.einsum('kj,ik->ij', At_br, D_bb)
O_br += 4 * np.einsum('ijkl,mj,ki->ml', vt_abar, D_bb, D_aa)
O_br += 4 * np.einsum('ijkl,mj,ki->ml', vt_abbr, D_bb, D_ba)
O_br -= 2 * np.einsum('ijkl,mi,kj->ml', vt_abar, D_ba, D_ab)
O_br -= 2 * np.einsum('ijkl,mi,kj->ml', vt_abbr, D_ba, D_bb)
O_br -= 2 * np.einsum('ijkl,mj,li->mk', vt_abra, D_bb, D_aa)
O_br -= 2 * np.einsum('ijkl,mj,li->mk', vt_abrb, D_bb, D_ba)
O_br += 4 * np.einsum('ijkl,mi,lj->mk', vt_abra, D_ba, D_ab)
O_br += 4 * np.einsum('ijkl,mi,lj->mk', vt_abrb, D_ba, D_bb)

double_mod_eri_abrs = 4 * np.einsum('ijkl,mi,nj->mnkl', vt_abrs, D_aa, D_bb)
double_mod_eri_abrs -= 2 * np.einsum('ijkl,mj,ni->mnkl', vt_abrs, D_ab, D_ba)
double_mod_eri_abrs -= 2 * np.einsum('ijlk,mi,nj->mnkl', vt_absr, D_aa, D_bb)
double_mod_eri_abrs += 4 * np.einsum('ijlk,mj,ni->mnkl', vt_absr, D_ab, D_ba)

G_abrs = ( 2 * np.einsum('ik,jl->ijkl', T_ar, O_bs) - np.einsum('il,jk->ijkl', T_as, O_br) +
          2 * np.einsum('jl,ik->ijkl', T_bs, O_ar) - np.einsum('jk,il->ijkl', T_br, O_as) + double_mod_eri_abrs)

v_rsab = sapt.v('rsab')
e_rsab = 1 / (-sapt.eps('r', dim=4) - sapt.eps('s', dim=3) + sapt.eps('a', dim=2) + sapt.eps('b'))
t_rsab = np.einsum('rsab,rsab->rsab', v_rsab, e_rsab)

e2_disp_full = np.einsum('ijkl,klij->', G_abrs, t_rsab)

# E20 Exchange-Dispersion
ExchDisp20 = e2_disp_full - Disp20

### Compare with Psi4
# Print Psi4Numpy Results
print('Psi4Numpy SAPT0 Results')
print('-' * 70)
sapt_printer('Elst10', Elst10)
sapt_printer('Exch10', Exch10)
sapt_printer('Disp20', Disp20)
sapt_printer('Exch-Disp20', ExchDisp20)
sapt_printer('Ind20,r', Ind20r)
sapt_printer('Exch-Ind20,r', ExchInd20r)
print('-' * 70)
sapt0_no_S2 = Exch10 + Elst10 + Disp20 + ExchDisp20 + Ind20r + ExchInd20r
sapt_printer('Total SAPT0', sapt0_no_S2)
print('')

# Print Psi4 Results
psi4.set_options({'df_basis_sapt': 'aug-cc-pvtz-ri'})
psi4.energy('sapt0')

Eelst = psi4.get_variable('SAPT ELST ENERGY')
Eexch = psi4.get_variable('SAPT EXCH10 ENERGY')
Eexch_S2 = psi4.get_variable('SAPT EXCH10(S^2) ENERGY')
Eind = psi4.get_variable('SAPT IND20,R ENERGY')
Eexind = psi4.get_variable('SAPT EXCH-IND20,R ENERGY')
Edisp = psi4.get_variable('SAPT DISP20 ENERGY')
Eexdisp = psi4.get_variable('SAPT EXCH-DISP20 ENERGY')

print('Psi4 SAPT0 Results')
print('-' * 70)
sapt_printer('Elst10', Eelst)
sapt_printer('Exch10', Eexch)
sapt_printer('Exch10(S^2)', Eexch_S2)
sapt_printer('Disp20', Edisp)
sapt_printer('Exch-Disp20(S^2)', Eexdisp)
sapt_printer('Ind20,r', Eind)
sapt_printer('Exch-Ind20,r(S^2)', Eexind)
print('-' * 70)
sapt0 = Eelst + Eexch_S2 + Edisp + Eexdisp + Eind + Eexind
sapt_printer('Total SAPT0(S^2)', sapt0)
print('')
