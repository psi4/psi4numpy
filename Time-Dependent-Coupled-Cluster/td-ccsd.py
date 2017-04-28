# A simple Psi4 script to compute TD-CCSD from a CCSD reference
# Scipy and numpy python modules are required
# This script runs a post-CCSD computation, using the ERI
# tensor, the T1 and T2 amplitudes as well as the F and W 
# intermediates.
#
# CCSD
# Created by: Daniel G. A. Smith
# Date: 7/29/14
# License: GPL v3.0
#
# TD-CCSD
# Created by: Daniel R. Nascimento
# Date: 4/22/17
# License: GPL v3.0
#

import time
import numpy as np
np.set_printoptions(precision=14, linewidth=200, suppress=True)
import psi4

# Set memory
psi4.set_memory(int(2e9), True)
psi4.core.set_output_file('output.dat', True)

numpy_memory = 2

mol = psi4.geometry("""
#O
#H 1 1.1
#H 1 1.1 2 104
#H 0.0 0.0 0.0
#H 1.0 0.0 0.0
#O          0.000000000000     0.000000000000    -0.068516219310
#H          0.000000000000    -0.790689573744     0.543701060724
#H          0.000000000000     0.790689573744     0.543701060724
C	0.0	0.0	0.0
O	0.0	0.0	1.0
symmetry c1
no_reorient
""")

psi4.set_options({'basis': 'sto-3g',
                  'scf_type': 'pk',
                  'mp2_type': 'conv',
                  'freeze_core': 'false',
                  'e_convergence': 1e-14,
                  'd_convergence': 1e-14})

# CCSD Settings
E_conv = 1.e-14
maxiter = 50
print_amps = False
compare_psi4 = False

# TD-CCSD Settings
### defines the field polarization (0 == x, 1 == y, 2 == z)
polarization = 1
### sets the total number of time steps
steps_total = 20000
### sets the length of the time step
time_step = 0.05

# First compute RHF energy using Psi4
scf_e, wfn = psi4.energy('SCF', return_wfn=True)

# Grab data from
C = wfn.Ca()
ndocc = wfn.doccpi()[0]
nmo = wfn.nmo()
SCF_E = wfn.energy()
eps = np.asarray(wfn.epsilon_a())

# Compute size of SO-ERI tensor in GB
ERI_Size = (nmo ** 4) * 128e-9
print('\nSize of the SO ERI tensor will be %4.2f GB.' % ERI_Size)
memory_footprint = ERI_Size * 5.2
if memory_footprint > numpy_memory:
    clean()
    raise Exception("Estimated memory utilization (%4.2f GB) exceeds numpy_memory \
                    limit of %4.2f GB." % (memory_footprint, numpy_memory))

# Integral generation from Psi4's MintsHelper
t = time.time()
mints = psi4.core.MintsHelper(wfn.basisset())
H = np.asarray(mints.ao_kinetic()) + np.asarray(mints.ao_potential())

print('\nTotal time taken for ERI integrals: %.3f seconds.\n' % (time.time() - t))

#Make spin-orbital MO
print('Starting AO -> spin-orbital MO transformation...')
t = time.time()
MO = np.asarray(mints.mo_spin_eri(C, C))

# Update nocc and nvirt
nso = nmo * 2
nocc = ndocc * 2
nvirt = nso - nocc

# Make slices
o = slice(0, nocc)
v = slice(nocc, MO.shape[0])

#Extend eigenvalues
eps = np.repeat(eps, 2)
Eocc = eps[o]
Evirt = eps[v]

print('..finished transformation in %.3f seconds.\n' % (time.time() - t))

# DPD approach to CCSD equations
# See: http://sirius.chem.vt.edu/wiki/doku.php?id=crawdad:programming

# occ orbitals i, j, k, l, m, n
# virt orbitals a, b, c, d, e, f
# all oribitals p, q, r, s, t, u, v


#Bulid Eqn 9: tilde{\Tau})
def build_tilde_tau(t1, t2):
    ttau = t2.copy()
    tmp = 0.5 * np.einsum('ia,jb->ijab', t1, t1)
    ttau += tmp
    ttau -= tmp.swapaxes(2, 3)
    return ttau


#Build Eqn 10: \Tau)
def build_tau(t1, t2):
    ttau = t2.copy()
    tmp = np.einsum('ia,jb->ijab', t1, t1)
    ttau += tmp
    ttau -= tmp.swapaxes(2, 3)
    return ttau


#Build Eqn 3:
def build_Fae(t1, t2):
    Fae = F[v, v].copy()
    Fae[np.diag_indices_from(Fae)] = 0

    Fae -= 0.5 * np.einsum('me,ma->ae', F[o, v], t1)
    Fae += np.einsum('mf,mafe->ae', t1, MO[o, v, v, v])

    tmp_tau = build_tilde_tau(t1, t2)
    Fae -= 0.5 * np.einsum('mnaf,mnef->ae', tmp_tau, MO[o, o, v, v])
    return Fae


#Build Eqn 4:
def build_Fmi(t1, t2):
    Fmi = F[o, o].copy()
    Fmi[np.diag_indices_from(Fmi)] = 0

    Fmi += 0.5 * np.einsum('ie,me->mi', t1, F[o, v])
    Fmi += np.einsum('ne,mnie->mi', t1, MO[o, o, o, v])

    tmp_tau = build_tilde_tau(t1, t2)
    Fmi += 0.5 * np.einsum('inef,mnef->mi', tmp_tau, MO[o, o, v, v])
    return Fmi


#Build Eqn 5:
def build_Fme(t1, t2):
    Fme = F[o, v].copy()
    Fme += np.einsum('nf,mnef->me', t1, MO[o, o, v, v])
    return Fme


#Build Eqn 6:
def build_Wmnij(t1, t2):
    Wmnij = MO[o, o, o, o].copy()

    Pij = np.einsum('je,mnie->mnij', t1, MO[o, o, o, v])
    Wmnij += Pij
    Wmnij -= Pij.swapaxes(2, 3)

    tmp_tau = build_tau(t1, t2)
    Wmnij += 0.25 * np.einsum('ijef,mnef->mnij', tmp_tau, MO[o, o, v, v])
    return Wmnij


#Build Eqn 7:
def build_Wabef(t1, t2):
    # Rate limiting step written using tensordot, ~10x faster
    # The commented out lines are consistent with the paper

    Wabef = MO[v, v, v, v].copy()

    Pab = np.einsum('baef->abef', np.tensordot(t1, MO[v, o, v, v], axes=(0, 1)))
    # Pab = np.einsum('mb,amef->abef', t1, MO[v, o, v, v])

    Wabef -= Pab
    Wabef += Pab.swapaxes(0, 1)

    tmp_tau = build_tau(t1, t2)

    Wabef += 0.25 * np.tensordot(tmp_tau, MO[v, v, o, o], axes=((0, 1), (2, 3)))
    # Wabef += 0.25 * np.einsum('mnab,mnef->abef', tmp_tau, MO[o, o, v, v])
    return Wabef


#Build Eqn 8:
def build_Wmbej(t1, t2):
    Wmbej = MO[o, v, v, o].copy()
    Wmbej += np.einsum('jf,mbef->mbej', t1, MO[o, v, v, v])
    Wmbej -= np.einsum('nb,mnej->mbej', t1, MO[o, o, v, o])

    tmp = (0.5 * t2) + np.einsum('jf,nb->jnfb', t1, t1)

    Wmbej -= np.einsum('jbme->mbej', np.tensordot(tmp, MO[o, o, v, v], axes=((1, 2), (1, 3))))
    # Wmbej -= np.einsum('jnfb,mnef->mbej', tmp, MO[o, o, v, v])
    return Wmbej


### Build so Fock matirx

# Update H, transform to MO basis and tile for alpha/beta spin
H = np.einsum('uj,vi,uv', C, C, H)
H = np.repeat(H, 2, axis=0)
H = np.repeat(H, 2, axis=1)

# Make H block diagonal
spin_ind = np.arange(H.shape[0], dtype=np.int) % 2
H *= (spin_ind.reshape(-1, 1) == spin_ind)

# Compute Fock matrix
F = H + np.einsum('pmqm->pq', MO[:, o, :, o])

### Build D matrices
Focc = F[np.arange(nocc), np.arange(nocc)].flatten()
Fvirt = F[np.arange(nocc, nvirt + nocc), np.arange(nocc, nvirt + nocc)].flatten()

Dia = Focc.reshape(-1, 1) - Fvirt
Dijab = Focc.reshape(-1, 1, 1, 1) + Focc.reshape(-1, 1, 1) - Fvirt.reshape(-1, 1) - Fvirt

### Construct initial guess

# t^a_i
t1 = np.zeros((nocc, nvirt))
# t^{ab}_{ij}
MOijab = MO[o, o, v, v]
t2 = MOijab / Dijab

### Compute MP2 in MO basis set to make sure the transformation was correct
MP2corr_E = np.einsum('ijab,ijab->', MOijab, t2) / 4
MP2_E = SCF_E + MP2corr_E

print('MO based MP2 correlation energy: %.8f' % MP2corr_E)
print('MP2 total energy:       %.8f' % MP2_E)
psi4.driver.p4util.compare_values(psi4.energy('mp2'), MP2_E, 6, 'MP2 Energy')

### Start CCSD iterations
print('\nStarting CCSD iterations')
ccsd_tstart = time.time()
CCSDcorr_E_old = 0.0
for CCSD_iter in range(1, maxiter + 1):
    ### Build intermediates
    Fae = build_Fae(t1, t2)
    Fmi = build_Fmi(t1, t2)
    Fme = build_Fme(t1, t2)

    Wmnij = build_Wmnij(t1, t2)
    Wabef = build_Wabef(t1, t2)
    Wmbej = build_Wmbej(t1, t2)

    #### Build RHS side of t1 equations
    rhs_T1  = F[o, v].copy()
    rhs_T1 += np.einsum('ie,ae->ia', t1, Fae)
    rhs_T1 -= np.einsum('ma,mi->ia', t1, Fmi)
    rhs_T1 += np.einsum('imae,me->ia', t2, Fme)
    rhs_T1 -= np.einsum('nf,naif->ia', t1, MO[o, v, o, v])
    rhs_T1 -= 0.5 * np.einsum('imef,maef->ia', t2, MO[o, v, v, v])
    rhs_T1 -= 0.5 * np.einsum('mnae,nmei->ia', t2, MO[o, o, v, o])

    ### Build RHS side of t2 equations
    rhs_T2 = MO[o, o, v, v].copy()

    # P_(ab) t_ijae (F_be - 0.5 t_mb F_me)
    tmp = Fae - 0.5 * np.einsum('mb,me->be', t1, Fme)
    Pab = np.einsum('ijae,be->ijab', t2, tmp)
    rhs_T2 += Pab
    rhs_T2 -= Pab.swapaxes(2, 3)

    # P_(ij) t_imab (F_mj + 0.5 t_je F_me)
    tmp = Fmi + 0.5 * np.einsum('je,me->mj', t1, Fme)
    Pij = np.einsum('imab,mj->ijab', t2, tmp)
    rhs_T2 -= Pij
    rhs_T2 += Pij.swapaxes(0, 1)

    tmp_tau = build_tau(t1, t2)
    rhs_T2 += 0.5 * np.einsum('mnab,mnij->ijab', tmp_tau, Wmnij)
    rhs_T2 += 0.5 * np.einsum('ijef,abef->ijab', tmp_tau, Wabef)

    # P_(ij) * P_(ab)
    # (ij - ji) * (ab - ba)
    # ijab - ijba -jiab + jiba
    tmp = np.einsum('ie,ma,mbej->ijab', t1, t1, MO[o, v, v, o])
    Pijab = np.einsum('imae,mbej->ijab', t2, Wmbej)
    Pijab -= tmp

    rhs_T2 += Pijab
    rhs_T2 -= Pijab.swapaxes(2, 3)
    rhs_T2 -= Pijab.swapaxes(0, 1)
    rhs_T2 += Pijab.swapaxes(0, 1).swapaxes(2, 3)

    Pij = np.einsum('ie,abej->ijab', t1, MO[v, v, v, o])
    rhs_T2 += Pij
    rhs_T2 -= Pij.swapaxes(0, 1)

    Pab = np.einsum('ma,mbij->ijab', t1, MO[o, v, o, o])
    rhs_T2 -= Pab
    rhs_T2 += Pab.swapaxes(2, 3)

    ### Update t1 and t2 amplitudes
    t1 = rhs_T1 / Dia
    t2 = rhs_T2 / Dijab

    ### Compute CCSD correlation energy
    CCSDcorr_E = np.einsum('ia,ia->', F[o, v], t1)
    CCSDcorr_E += 0.25 * np.einsum('ijab,ijab->', MO[o, o, v, v], t2)
    CCSDcorr_E += 0.5 * np.einsum('ijab,ia,jb->', MO[o, o, v, v], t1, t1)

    ### Print CCSD correlation energy
    print('CCSD Iteration %3d: CCSD correlation = %3.12f  '\
          'dE = %3.5E' % (CCSD_iter, CCSDcorr_E, (CCSDcorr_E - CCSDcorr_E_old)))
    if (abs(CCSDcorr_E - CCSDcorr_E_old) < E_conv):
        break

    CCSDcorr_E_old = CCSDcorr_E

print('CCSD iterations took %.2f seconds.\n' % (time.time() - ccsd_tstart))

CCSD_E = SCF_E + CCSDcorr_E

print('\nFinal CCSD correlation energy:     % 16.10f' % CCSDcorr_E)
print('Total CCSD energy:                 % 16.10f' % CCSD_E)
if compare_psi4:
    psi4.driver.p4util.compare_values(psi4.energy('CCSD'), CCSD_E, 6, 'CCSD Energy')

if print_amps:
    # [::4] take every 4th, [-5:] take last 5, [::-1] reverse order
    t2_args = np.abs(t2).ravel().argsort()[::2][-5:][::-1]
    t1_args = np.abs(t1).ravel().argsort()[::4][-5:][::-1]

    print('\nLargest t1 amplitudes')
    for pos in t1_args:
        value = t1.flat[pos]
        inds = np.unravel_index(pos, t1.shape)
        print('%4d  %4d |   % 5.10f' % (inds[0], inds[1], value))

    print('\nLargest t2 amplitudes')
    for pos in t2_args:
        value = t2.flat[pos]
        inds = np.unravel_index(pos, t2.shape)


### TD-CCSD code starts here!!!

#Step 1: Build one and two-particle elements of the similarity-transformed Hamiltonian
#Equations from J. Gauss and J.F. Stanton, JCP, 103, 9, 1995. Table III (b)

#### Note that Eqs. 1 - 6 only modifies some intermediates that were already defined during 
#### the ground-state CCSD computation!

# 1st equation
def update_Fae(t1, Fae, Fme):
    Fae -= 0.5 * np.einsum('me,ma->ae', Fme, t1)
    return Fae

# 2nd equation
def update_Fmi(t1, Fmi, Fme):
    Fmi += 0.5 * np.einsum('me,ie->mi', Fme, t1)
    return Fmi

# 4th equation
def update_Wmnij(t1, t2, Wmnij):
    tmp_tau = build_tau(t1, t2)

    Wmnij += 0.25 * np.einsum('ijef,mnef->mnij', tmp_tau, MO[o, o, v, v])
    return Wmnij

# 5th equation
def update_Wabef(t1, t2, Wabef):
    tmp_tau = build_tau(t1, t2)

    Wabef += 0.25 * np.einsum('mnab,mnef->abef', tmp_tau, MO[o, o, v, v])
    return Wabef

# 6th equation
def update_Wmbej(t2, Wmbej):
    Wmbej -= 0.5 * np.einsum('jnfb,mnef->mbej', t2, MO[o, o, v, v])
    return Wmbej

#### Eqs. 7 - 10 define new intermediates

# 7th equation
def build_Wmnie(t1):
    Wmnie = MO[o, o, o, v].copy()
    Wmnie += np.einsum('if,mnfe->mnie', t1, MO[o, o, v, v])
    return Wmnie

# 8th equation
def build_Wamef(t1):
    Wamef = MO[v, o, v, v].copy()
    Wamef -= np.einsum('na,nmef->amef', t1, MO[o, o, v, v])
    return Wamef

# 9th equation
def build_Wmbij(t1, t2, Fme, Wmnij):
    Wmbij = MO[o, v, o, o].copy()
    
    Wmbij -= np.einsum('me,ijbe->mbij', Fme, t2)
    Wmbij -= np.einsum('nb,mnij->mbij', t1, Wmnij)
   
    temp_tau = build_tau(t1, t2) 
    Wmbij += 0.5 * np.einsum('mbef,ijef->mbij', MO[o, v, v, v],temp_tau)
   
    Pij = np.einsum('jnbe,mnie->mbij', t2, MO[o, o, o, v])
    Wmbij += Pij
    Wmbij -= Pij.swapaxes(2, 3)

    temp_mbej = MO[o, v, v, o].copy()
    temp_mbej -= np.einsum('njbf,mnef->mbej', t2, MO[o, o, v, v])
    Pij = np.einsum('ie,mbej->mbij', t1, temp_mbej)
    Wmbij += Pij
    Wmbij -= Pij.swapaxes(2, 3)
    return Wmbij

# 10th equation
def build_Wabei(t1, t2, Fme, Wabef):
    Wabei = MO[v, v, v, o].copy()

    Wabei -= np.einsum('me,miab->abei', Fme, t2)
    Wabei += np.einsum('if,abef->abei', t1, Wabef)
    
    temp_tau = build_tau(t1, t2) 
    Wabei += 0.5 * np.einsum('mnei,mnab->abei', MO[o, o, v, o],temp_tau)
   
    Pab = np.einsum('mbef,miaf->abei', MO[o, v, v, v], t2)
    Wabei -= Pab
    Wabei += Pab.swapaxes(0, 1)

    temp_mbei = MO[o, v, v, o].copy()
    temp_mbei -= np.einsum('nibf,mnef->mbei', t2, MO[o, o, v, v])
    Pab = np.einsum('ma,mbei->abei', t1, temp_mbei)
    Wabei -= Pab
    Wabei += Pab.swapaxes(0, 1)
    return Wabei    
 
### Build three-body intermediates: Table III (c)

# 1st equation
def build_Gae(t2, l2):
    Gae = -0.5 * np.einsum('mnef,mnaf->ae', t2, l2)
    return Gae

# 2nd equation
def build_Gmi(t2, l2):
    Gmi = 0.5 * np.einsum('mnef,inef->mi', t2, l2)
    return Gmi

### Construct initial guess for lambda amplitudes
l1 = np.zeros((nocc, nvirt))
l2 = t2.copy()

### Update/build intermediates that do not depende on lambda
Fae   = update_Fae(t1, Fae, Fme)
Fmi   = update_Fmi(t1, Fmi, Fme)
Wmnij = update_Wmnij(t1, t2, Wmnij)
Wabef = update_Wabef(t1, t2, Wabef)
Wmbej = update_Wmbej(t2, Wmbej)

Wmnie = build_Wmnie(t1)
Wamef = build_Wamef(t1)
Wmbij = build_Wmbij(t1, t2, Fme, Wmnij)
Wabei = build_Wabei(t1, t2, Fme, Wabef)

### begin LCCSD iterations: The lambda equations are in Table II. 
print('\nStarting LCCSD iterations')
lccsd_tstart = time.time()
LCCSDcorr_E_old = 0.0
for LCCSD_iter in range(1, maxiter + 1):

    # Build intermediates that depend on lambda
    Gae = build_Gae(t2, l2)
    Gmi = build_Gmi(t2, l2)

    # Build RHS of l1 equations: Table II (a)
    rhs_L1  = Fme.copy()
    rhs_L1 += np.einsum('ie,ea->ia', l1, Fae)
    rhs_L1 -= np.einsum('ma,im->ia', l1, Fmi)
    rhs_L1 += np.einsum('me,ieam->ia', l1, Wmbej)
    rhs_L1 += 0.5 * np.einsum('imef,efam->ia', l2, Wabei)
    rhs_L1 -= 0.5 * np.einsum('mnae,iemn->ia', l2, Wmbij)
    rhs_L1 -= np.einsum('ef,eifa->ia', Gae, Wamef)
    rhs_L1 -= np.einsum('mn,mina->ia', Gmi, Wmnie)

    ### Build RHS of l2 equations
    ### Table II (b)
    rhs_L2 = MO[o, o, v, v].copy()

    # P_(ab) l_ijae * F_eb
    Pab = np.einsum('ijae,eb->ijab', l2, Fae)
    rhs_L2 += Pab
    rhs_L2 -= Pab.swapaxes(2, 3)

    # P_(ij) l_imab * F_jm
    Pij = np.einsum('imab,jm->ijab', l2, Fmi)
    rhs_L2 -= Pij
    rhs_L2 += Pij.swapaxes(0, 1)

    # 0.5 * l_mnab * W_ijmn
    rhs_L2 += 0.5 * np.einsum('mnab,ijmn->ijab', l2, Wmnij)

    # 0.5 * l_ijef * W_efab
    rhs_L2 += 0.5 * np.einsum('ijef,efab->ijab', l2, Wabef)

    # P_(ij) l_ie W_ejab
    Pij = np.einsum('ie,ejab->ijab', l1, Wamef)
    rhs_L2 += Pij
    rhs_L2 -= Pij.swapaxes(0, 1)

    # P_(ab) l_ma W_ijmb
    Pab = np.einsum('ma,ijmb->ijab', l1, Wmnie)
    rhs_L2 -= Pab
    rhs_L2 += Pab.swapaxes(2, 3)
   
    # P_(ij) P_(ab) l_imae W_jebm
    Pijab = np.einsum('imae,jebm->ijab', l2, Wmbej)
    rhs_L2 += Pijab
    rhs_L2 -= Pijab.swapaxes(0, 1)
    rhs_L2 -= Pijab.swapaxes(2, 3)
    rhs_L2 += Pijab.swapaxes(0, 1).swapaxes(2, 3)
     
    # P_(ij) P_(ab) l_ia F_jb
    Pijab = np.einsum('ia,jb->ijab', l1, Fme)
    rhs_L2 += Pijab
    rhs_L2 -= Pijab.swapaxes(0, 1)
    rhs_L2 -= Pijab.swapaxes(2, 3)
    rhs_L2 += Pijab.swapaxes(0, 1).swapaxes(2, 3)
    
    # P_(ab) <ij||ae> G_be
    Pab = np.einsum('be,ijae->ijab', Gae, MO[o, o, v, v])
    rhs_L2 += Pab
    rhs_L2 -= Pab.swapaxes(2, 3)

    # P_(ij) <im||ab> G_mj
    Pij = np.einsum('mj,imab->ijab', Gmi, MO[o, o, v, v])
    rhs_L2 -= Pij
    rhs_L2 += Pij.swapaxes(0, 1)

    # Update l1 and l2 amplitudes
    l1 = rhs_L1 / Dia
    l2 = rhs_L2 / Dijab
    
    # Compute LCCSD pseudoenergy 
    # E = sum_{ia} l_a^i f_{ia} + 1/4 sum_{ijab} l_ab^ij <ij||ab>
    LCCSDcorr_E = np.einsum('ia,ia->', F[o, v], l1)
    LCCSDcorr_E += 0.25 * np.einsum('ijab,ijab->', MO[o, o, v, v], l2)

    # Print LCCSD pseudoenergy
    print('LCCSD Iteration %3d: LCCSD pseudoenergy = %3.12f  '\
          'dE = %3.5E' % (LCCSD_iter, LCCSDcorr_E, (LCCSDcorr_E - LCCSDcorr_E_old)))
    if (abs(LCCSDcorr_E - LCCSDcorr_E_old) < E_conv):
        break

    LCCSDcorr_E_old = LCCSDcorr_E

print('LCCSD iterations took %.2f seconds.\n' % (time.time() - lccsd_tstart))

LCCSD_E = SCF_E + LCCSDcorr_E

print('\nFinal LCCSD correlation energy:     % 16.10f' % LCCSDcorr_E)
print('Total LCCSD energy:                 % 16.10f' % LCCSD_E)

# Step 2: Build left and right dipole functions for t = 0 
# Equations from D. R. Nascimento and A. E. DePrince III, JCTC, 12, 5834 (2016)

print('\nBuilding dipole functions')

### Grab dipole integrals and transform to MO basis

# mu in the AO basis
mu = np.asarray(mints.ao_dipole()[polarization])
# AO -> MO transformation
mu = np.einsum('uj,vi,uv', C, C, mu)
mu = np.repeat(mu, 2, axis=0)
mu = np.repeat(mu, 2, axis=1)
spin_ind = np.arange(mu.shape[0], dtype=np.int) % 2
mu *= (spin_ind.reshape(-1, 1) == spin_ind)

### Build dipole functions
dipole_build_tstart = time.time()

# Right dipole function
# In the text below Eq. 18
trace = np.trace(mu[o , o])
mr0 = trace
mr1 = mu[o, v].copy()
mr2 = l2*0.0

# Left dipole function

# Eq. 19
ml0  = trace
ml0 += np.einsum('ia,ia->', mu[o, v], l1)

# Eq. 20
ml1  = mu[o, v].copy()
ml1 += trace * l1 
ml1 += np.einsum('ea,ie->ia', mu[v, v], l1)
ml1 -= np.einsum('im,ma->ia', mu[o, o], l1)
ml1 += np.einsum('imae,em->ia', l2, mu[v, o])

# Eq. 21
ml2  = np.einsum('ia,jb->ijab', l1, mu[o, v])
ml2 -= np.einsum('ib,ja->ijab', l1, mu[o, v])
ml2 += np.einsum('jb,ia->ijab', l1, mu[o, v])
ml2 -= np.einsum('ja,ib->ijab', l1, mu[o, v])

ml2 += trace * l2

Pab = np.einsum('ijeb,ea->ijab', l2, mu[v, v])
ml2 += Pab
ml2 -= Pab.swapaxes(2, 3)

Pij = np.einsum('im,mjab->ijab', mu[o, o], l2)
ml2 -= Pij
ml2 += Pij.swapaxes(0, 1)

print('Dipole function build took %.2f seconds.\n' % (time.time() - dipole_build_tstart))

# Step 3: Build the time evolution of the dipole function (right only!)
# Equations from D. R. Nascimento and A. E. DePrince III, JCTC, 12, 5834 (2016)

#### Important: Update Fae and Fmi so that they contain the diagonals of the
#### Fock matrix.
#### Fae += f_{ae} and Fmi += f_{mi}
Fae += F[v, v]
Fmi += F[o, o]

# Eq. 24 
def compute_dmr0(Mr1, Mr2):
    dMr0  = np.einsum('ia,ia->',Mr1 , Fme)
    dMr0 += 0.25 * np.einsum('ijab,ijab->',Mr2, MO[o, o, v, v])
    return -1j * dMr0

# Eq. 25
def compute_dmr1(Mr1, Mr2):
    dMr1  = np.einsum('ib,ab->ia',Mr1 , Fae)
    dMr1 -= np.einsum('ji,ja->ia',Fmi , Mr1)
    dMr1 += np.einsum('jb,jabi->ia',Mr1 , Wmbej)
    dMr1 += np.einsum('ijab,jb->ia',Mr2 , Fme)
    dMr1 -= 0.5 * np.einsum('jkib,jkab->ia',Wmnie , Mr2)
    dMr1 += 0.5 * np.einsum('ijbc,ajbc->ia',Mr2 , Wamef)
    return -1j * dMr1

# Eq. 26 (the paper only contains the terms for CC2, here we provide the 
# additional terms for CCSD)
def compute_dmr2(Mr1, Mr2):
    # P_(ab) M_m^b W_{mbij}
    Pab = np.einsum('mb,maij->ijab', Mr1, Wmbij)
    dMr2  = Pab
    dMr2 -= Pab.swapaxes(2, 3)

    # + P_(ij) [M_n^e W_{mnie}] t_{jm}^{ab}
    temp = np.einsum('ne,mnie->im', Mr1, Wmnie)
    Pij = np.einsum('im,jmab->ijab', temp, t2)
    dMr2 += Pij
    dMr2 -= Pij.swapaxes(0, 1)

    # - P_(ij) M_j^e W_{abei}
    Pij = np.einsum('je,abei->ijab', Mr1, Wabei)
    dMr2 -= Pij
    dMr2 += Pij.swapaxes(0, 1)

    # - P_(ab) [M_m^f W_{amef}] t_{ij}^{be}
    temp = -1.0 * np.einsum('mf,amef->ae', Mr1, Wamef)
    Pab = np.einsum('ae,ijbe->ijab', temp, t2)
    dMr2 += Pab
    dMr2 -= Pab.swapaxes(2, 3)

    # + P_(ij) M_{jm}^{ab} F_{mi}
    Pij = np.einsum('jmab,mi->ijab', Mr2, Fmi)
    dMr2 += Pij
    dMr2 -= Pij.swapaxes(0, 1)
    
    # + 0.5 * P_(ij) [M_{in}^{ef} <mn||ef>] t_{jm}^{ab}
    temp = 0.5 * np.einsum('inef,mnef->im', Mr2, MO[o, o, v, v])
    Pij = np.einsum('im,jmab->ijab', temp, t2)
    dMr2 += Pij
    dMr2 -= Pij.swapaxes(0, 1)

    # + P_(ab) M_{ij}^{ae} F_{be}
    Pab = np.einsum('ijae,be->ijab', Mr2, Fae)
    dMr2 += Pab
    dMr2 -= Pab.swapaxes(2, 3)

    # + 0.5 * P_(ab) [M_{mn}^{af} <mn||ef>] t_{ij}^{be}
    temp = 0.5 * np.einsum('mnaf,mnef->ae', Mr2, MO[o, o, v, v])
    Pab = np.einsum('ae,ijbe->ijab',temp , t2)
    dMr2 += Pab
    dMr2 -= Pab.swapaxes(2, 3)

    # + P_(ij) P_(ab) M_{mj}^{ae} W_{mbei}
    Pijab = np.einsum('mjae,mbei->ijab', Mr2, Wmbej)
    dMr2 += Pijab
    dMr2 -= Pijab.swapaxes(0, 1)
    dMr2 -= Pijab.swapaxes(2, 3)
    dMr2 += Pijab.swapaxes(0, 1).swapaxes(2, 3)

    # + 0.5 * M_{mn}^{ab} W_{mnij}
    dMr2 += 0.5 * np.einsum('mnab,mnij->ijab', Mr2, Wmnij)

    # + 0.5 * M_{ij}^{ef} W_{abef}
    dMr2 += 0.5 * np.einsum('ijef,abef->ijab', Mr2, Wabef)

    return -1j * dMr2    

# Step 4: Time propagation. 
# Equations from D. R. Nascimento and A. E. DePrince III, JCTC, 12, 5834 (2016)

# Here we will use the 4th-order Runge-Kutta scheme.
# y_{n+1} = y_n + (h/6) * (k_1 + 2k_2 + 2k_3 +k_4)
# k1 = f(t_n, y_n)
# k2 = f(t_n + h/2, y_n+ h*k_{1}/2)
# k3 = f(t_n + h/2, y_n+ h*k_{2}/2)
# k4 = f(t_n + h, y_n+ h*k_3)

# A nice overview of this scheme can be found at
# http://lpsa.swarthmore.edu/NumInt/NumIntFourth.html

### Time propagation

# Initialize complex dipole function
M0 = mr0 + 1j * 0
M1 = mr1 + 1j * 0
M2 = mr2 + 1j * 0

# Begin propagation 

print('Starting time propagation\n')
print('                 Time            Re{<M(0)|M(t)>}      Im{<M(0)|M(t)>}')

propagation_tstart = time.time()
for step in range(0, steps_total + 1):

    curtime = step * time_step

    # compute k1
    k1_0 = compute_dmr0(M1, M2)
    k1_1 = compute_dmr1(M1, M2)
    k1_2 = compute_dmr2(M1, M2)

    temp_0 =  M0 + 0.5 * time_step * k1_0
    temp_1 =  M1 + 0.5 * time_step * k1_1
    temp_2 =  M2 + 0.5 * time_step * k1_2

    # compute k2
    k2_0 = compute_dmr0(temp_1, temp_2) 
    k2_1 = compute_dmr1(temp_1, temp_2) 
    k2_2 = compute_dmr2(temp_1, temp_2) 

    temp_0 =  M0 + 0.5 * time_step * k2_0
    temp_1 =  M1 + 0.5 * time_step * k2_1
    temp_2 =  M2 + 0.5 * time_step * k2_2

    # compute k3
    k3_0 = compute_dmr0(temp_1, temp_2) 
    k3_1 = compute_dmr1(temp_1, temp_2) 
    k3_2 = compute_dmr2(temp_1, temp_2) 

    temp_0 = M0 + 1.0 * time_step * k3_0
    temp_1 = M1 + 1.0 * time_step * k3_1
    temp_2 = M2 + 1.0 * time_step * k3_2

    # compute k4
    k4_0 = compute_dmr0(temp_1, temp_2) 
    k4_1 = compute_dmr1(temp_1, temp_2) 
    k4_2 = compute_dmr2(temp_1, temp_2) 

    # compute dipole function at time t_0 + time_step   
    M0 += (time_step/6.0) * (k1_0 + 2.0 * k2_0 + 2.0 * k3_0 + 1.0 * k4_0)
    M1 += (time_step/6.0) * (k1_1 + 2.0 * k2_1 + 2.0 * k3_1 + 1.0 * k4_1)
    M2 += (time_step/6.0) * (k1_2 + 2.0 * k2_2 + 2.0 * k3_2 + 1.0 * k4_2)

    # compute autocorrelation function <M(0)|M(t)>
    # Eq. 13
    corr_func  = ml0 * M0
    corr_func += np.einsum('ia,ia->', ml1, M1)
    corr_func += 0.25 * np.einsum('ijab,ijab->', ml2, M2)

    print('@TIME %20.12f %20.12f %20.12f' % (curtime, corr_func.real, corr_func.imag))

print('Time-propagation took %.2f seconds.\n' % (time.time() - propagation_tstart))
print('Computation finished!\n')
