"""
Script to compute TD-CCSD from a CCSD reference.

This script runs a post-CCSD computation, using the ERI
tensor, the T1 and T2 amplitudes as well as the F and W 
intermediates.

References:
- DPD formulation of CC: [Stanton:1991:4334]
- TD-CCSD equations & algorithms: [Nascimento:2016]
"""

__authors__   =  "Daniel R. Nascimento"
__credits__   =  ["Daniel R. Nascimento"]

__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__   = "BSD-3-Clause"
__date__      = "2017-04-22"

import time
import numpy as np
from helper_CC import *
np.set_printoptions(precision=14, linewidth=200, suppress=True)
import psi4

# Set memory
psi4.set_memory(int(2e9), True)
psi4.core.set_output_file('output.dat', True)

numpy_memory = 2

mol = psi4.geometry("""
C	0.0	0.0	0.0
O	0.0	0.0	1.0
symmetry c1
no_reorient
""")

psi4.set_options({'basis': 'sto-3g',
                  'scf_type': 'pk',
                  'mp2_type': 'conv',
                  'freeze_core': 'false',
                  'e_convergence': 1e-6,
                  'd_convergence': 1e-6})

# LCCSD Settings
E_conv = 1.e-6
maxiter = 50

# TD-CCSD Settings
### defines the field polarization (0 == x, 1 == y, 2 == z)
polarization = 1
### sets the total number of time steps
steps_total = 20000
### sets the length of the time step
time_step = 0.05

# Compute CCSD
ccsd = helper_CCSD(mol, memory=2)
ccsd.compute_energy()

# Grab T amplitudes from CCSD
t1 = ccsd.t1
t2 = ccsd.t2

# Make slices
o = ccsd.slice_o
v = ccsd.slice_v

#Step 1: Build one and two-particle elements of the similarity-transformed Hamiltonian
#Equations from [Gauss:1995:3561], Table III (b) & (c)

#### Note that Eqs. 1 - 6 of Table III (b) only modifies some intermediates that were 
#### already defined during the ground-state CCSD computation!

# 3rd equation
Fme   = ccsd.build_Fme()

# 1st equation
def update_Fae():
    """Eqn 1 from [Gauss:1995:3561], Table III (b)"""
    Fae  = ccsd.build_Fae()
    Fae -= 0.5 * np.einsum('me,ma->ae', Fme, t1)
    return Fae

# 2nd equation
def update_Fmi():
    """Eqn 2 from [Gauss:1995:3561], Table III (b)"""
    Fmi  = ccsd.build_Fmi()
    Fmi += 0.5 * np.einsum('me,ie->mi', Fme, t1)
    return Fmi

# 4th equation
def update_Wmnij():
    """Eqn 4 from [Gauss:1995:3561], Table III (b)"""
    tmp_tau = ccsd.build_tau()

    Wmnij  = ccsd.build_Wmnij()
    Wmnij += 0.25 * np.einsum('ijef,mnef->mnij', tmp_tau, ccsd.get_MO('oovv'))
    return Wmnij

# 5th equation
def update_Wabef():
    """Eqn 5 from [Gauss:1995:3561], Table III (b)"""
    tmp_tau = ccsd.build_tau()

    Wabef  = ccsd.build_Wabef()
    Wabef += 0.25 * np.einsum('mnab,mnef->abef', tmp_tau, ccsd.get_MO('oovv'))
    return Wabef

# 6th equation
def update_Wmbej():
    """Eqn 6 from [Gauss:1995:3561], Table III (b)"""
    Wmbej  = ccsd.build_Wmbej()
    Wmbej -= 0.5 * np.einsum('jnfb,mnef->mbej', t2, ccsd.get_MO('oovv'))
    return Wmbej

#### Eqs. 7 - 10 define new intermediates

# 7th equation
def build_Wmnie():
    """Eqn 7 from [Gauss:1995:3561], Table III (b)"""
    Wmnie = ccsd.get_MO('ooov').copy()
    Wmnie += np.einsum('if,mnfe->mnie', t1, ccsd.get_MO('oovv'))
    return Wmnie

# 8th equation
def build_Wamef():
    """Eqn 8 from [Gauss:1995:3561], Table III (b)"""
    Wamef = ccsd.get_MO('vovv').copy()
    Wamef -= np.einsum('na,nmef->amef', t1, ccsd.get_MO('oovv'))
    return Wamef

# 9th equation
def build_Wmbij():
    """Eqn 9 from [Gauss:1995:3561], Table III (b)"""
    Wmbij = ccsd.get_MO('ovoo').copy()
    
    Wmbij -= np.einsum('me,ijbe->mbij', Fme, t2)
    Wmbij -= np.einsum('nb,mnij->mbij', t1, Wmnij)
   
    temp_tau = ccsd.build_tau() 
    Wmbij += 0.5 * np.einsum('mbef,ijef->mbij', ccsd.get_MO('ovvv'),temp_tau)
   
    Pij = np.einsum('jnbe,mnie->mbij', t2, ccsd.get_MO('ooov'))
    Wmbij += Pij
    Wmbij -= Pij.swapaxes(2, 3)

    temp_mbej = ccsd.get_MO('ovvo').copy()
    temp_mbej -= np.einsum('njbf,mnef->mbej', t2, ccsd.get_MO('oovv'))
    Pij = np.einsum('ie,mbej->mbij', t1, temp_mbej)
    Wmbij += Pij
    Wmbij -= Pij.swapaxes(2, 3)
    return Wmbij

# 10th equation
def build_Wabei():
    """Eqn 10 from [Gauss:1995:3561], Table III (b)"""
    Wabei = ccsd.get_MO('vvvo').copy()

    Wabei -= np.einsum('me,miab->abei', Fme, t2)
    Wabei += np.einsum('if,abef->abei', t1, Wabef)
    
    temp_tau = ccsd.build_tau() 
    Wabei += 0.5 * np.einsum('mnei,mnab->abei', ccsd.get_MO('oovo'),temp_tau)
   
    Pab = np.einsum('mbef,miaf->abei', ccsd.get_MO('ovvv'), t2)
    Wabei -= Pab
    Wabei += Pab.swapaxes(0, 1)

    temp_mbei = ccsd.get_MO('ovvo').copy()
    temp_mbei -= np.einsum('nibf,mnef->mbei', t2, ccsd.get_MO('oovv'))
    Pab = np.einsum('ma,mbei->abei', t1, temp_mbei)
    Wabei -= Pab
    Wabei += Pab.swapaxes(0, 1)
    return Wabei    
 
### Build three-body intermediates: [Gauss:1995:3561] Table III (c)

# 1st equation
def build_Gae(t2, l2):
    """Eqn 1 from [Gauss:1995:3561], Table III (c)"""
    Gae = -0.5 * np.einsum('mnef,mnaf->ae', t2, l2)
    return Gae

# 2nd equation
def build_Gmi(t2, l2):
    """Eqn 2 from [Gauss:1995:3561], Table III (c)"""
    Gmi = 0.5 * np.einsum('mnef,inef->mi', t2, l2)
    return Gmi

### Construct initial guess for lambda amplitudes
l1 = t1*0
l2 = t2.copy()

### Update/build intermediates that do not depende on lambda
Fae   = update_Fae()
Fmi   = update_Fmi()
Wmnij = update_Wmnij()
Wabef = update_Wabef()
Wmbej = update_Wmbej()

Wmnie = build_Wmnie()
Wamef = build_Wamef()
Wmbij = build_Wmbij()
Wabei = build_Wabei()

### begin LCCSD iterations: Lambda equations from [Gauss:1995:3561] Table II, (a) & (b). 
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
    rhs_L2 = ccsd.get_MO('oovv').copy()

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
    Pab = np.einsum('be,ijae->ijab', Gae, ccsd.get_MO('oovv'))
    rhs_L2 += Pab
    rhs_L2 -= Pab.swapaxes(2, 3)

    # P_(ij) <im||ab> G_mj
    Pij = np.einsum('mj,imab->ijab', Gmi, ccsd.get_MO('oovv'))
    rhs_L2 -= Pij
    rhs_L2 += Pij.swapaxes(0, 1)

    # Update l1 and l2 amplitudes
    l1 = rhs_L1 / ccsd.Dia
    l2 = rhs_L2 / ccsd.Dijab
    
    # Compute LCCSD pseudoenergy 
    # E = sum_{ia} l_a^i f_{ia} + 1/4 sum_{ijab} l_ab^ij <ij||ab>
    LCCSDcorr_E = np.einsum('ia,ia->', ccsd.get_F('ov'), l1)
    LCCSDcorr_E += 0.25 * np.einsum('ijab,ijab->', ccsd.get_MO('oovv'), l2)

    # Print LCCSD pseudoenergy
    print('LCCSD Iteration %3d: LCCSD pseudoenergy = %3.12f  '\
          'dE = %3.5E' % (LCCSD_iter, LCCSDcorr_E, (LCCSDcorr_E - LCCSDcorr_E_old)))
    if (abs(LCCSDcorr_E - LCCSDcorr_E_old) < E_conv):
        break

    LCCSDcorr_E_old = LCCSDcorr_E

print('LCCSD iterations took %.2f seconds.\n' % (time.time() - lccsd_tstart))

LCCSD_E = ccsd.rhf_e + LCCSDcorr_E

print('\nFinal LCCSD correlation energy:     % 16.10f' % LCCSDcorr_E)
print('Total LCCSD energy:                 % 16.10f' % LCCSD_E)

# Step 2: Build left and right dipole functions for t = 0 
# Equations from [Nascimento:2016:5834]

print('\nBuilding dipole functions')

### Grab dipole integrals and transform to MO basis

# mu in the AO basis
mints = psi4.core.MintsHelper(ccsd.wfn.basisset())
mu = np.asarray(mints.ao_dipole()[polarization])
# AO -> MO transformation
mu = np.einsum('uj,vi,uv', ccsd.C, ccsd.C, mu)
mu = np.repeat(mu, 2, axis=0)
mu = np.repeat(mu, 2, axis=1)
spin_ind = np.arange(mu.shape[0], dtype=np.int) % 2
mu *= (spin_ind.reshape(-1, 1) == spin_ind)

### Build dipole functions
dipole_build_tstart = time.time()

# Right dipole function
# [Nascimento:2016:5834], in the text below Eqn. 18
trace = np.trace(mu[o , o])
mr0 = trace
mr1 = mu[o, v].copy()
mr2 = l2*0.0

# Left dipole function

# [Nascimento:2016:5834], Eqn. 19
ml0  = trace
ml0 += np.einsum('ia,ia->', mu[o, v], l1)

# [Nascimento:2016:5834], Eqn. 20
ml1  = mu[o, v].copy()
ml1 += trace * l1 
ml1 += np.einsum('ea,ie->ia', mu[v, v], l1)
ml1 -= np.einsum('im,ma->ia', mu[o, o], l1)
ml1 += np.einsum('imae,em->ia', l2, mu[v, o])

# [Nascimento:2016:5834], Eqn. 21
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
# Equations from [Nascimento:2016:5834]

#### Important: Update Fae and Fmi so that they contain the diagonals of the
#### Fock matrix.
#### Fae += f_{aa} and Fmi += f_{ii}
Fae += ccsd.get_F('vv')
Fmi += ccsd.get_F('oo')

# [Nascimento:2016:5834], Eqn. 24 
def compute_dmr0(Mr1, Mr2):
    """Computes [Nascimento:2016:5834], Eqn. 24"""
    dMr0  = np.einsum('ia,ia->',Mr1 , Fme)
    dMr0 += 0.25 * np.einsum('ijab,ijab->',Mr2, ccsd.get_MO('oovv'))
    return -1j * dMr0

# [Nascimento:2016:5834], Eqn. 25
def compute_dmr1(Mr1, Mr2):
    """Computes [Nascimento:2016:5834], Eqn. 25"""
    dMr1  = np.einsum('ib,ab->ia',Mr1 , Fae)
    dMr1 -= np.einsum('ji,ja->ia',Fmi , Mr1)
    dMr1 += np.einsum('jb,jabi->ia',Mr1 , Wmbej)
    dMr1 += np.einsum('ijab,jb->ia',Mr2 , Fme)
    dMr1 -= 0.5 * np.einsum('jkib,jkab->ia',Wmnie , Mr2)
    dMr1 += 0.5 * np.einsum('ijbc,ajbc->ia',Mr2 , Wamef)
    return -1j * dMr1

# [Nascimento:2016:5834], Eqn. 26 
# (the paper only contains the terms for CC2, here we provide the 
# additional terms for CCSD)
def compute_dmr2(Mr1, Mr2):
    """Computes [Nascimento:2016:5834], Eqn. 26

    The paper above only contains terms for CC2; here we provide the
    additional terms for CCSD.
    """
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
    temp = 0.5 * np.einsum('inef,mnef->im', Mr2, ccsd.get_MO('oovv'))
    Pij = np.einsum('im,jmab->ijab', temp, t2)
    dMr2 += Pij
    dMr2 -= Pij.swapaxes(0, 1)

    # + P_(ab) M_{ij}^{ae} F_{be}
    Pab = np.einsum('ijae,be->ijab', Mr2, Fae)
    dMr2 += Pab
    dMr2 -= Pab.swapaxes(2, 3)

    # + 0.5 * P_(ab) [M_{mn}^{af} <mn||ef>] t_{ij}^{be}
    temp = 0.5 * np.einsum('mnaf,mnef->ae', Mr2, ccsd.get_MO('oovv'))
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
# Equations from [Nascimento:2016:5834] & [Cheever:2015]

# Here we will use the 4th-order Runge-Kutta scheme.
# See "Fourth" tab of [Cheever:2015] for details.

# y_{n+1} = y_n + (h/6) * (k_1 + 2k_2 + 2k_3 +k_4)
# k1 = f(t_n, y_n)
# k2 = f(t_n + h/2, y_n+ h*k_{1}/2)
# k3 = f(t_n + h/2, y_n+ h*k_{2}/2)
# k4 = f(t_n + h, y_n+ h*k_3)

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
    # [Nascimento:2016:5834], Eqn. 13
    corr_func  = ml0 * M0
    corr_func += np.einsum('ia,ia->', ml1, M1)
    corr_func += 0.25 * np.einsum('ijab,ijab->', ml2, M2)

    print('@TIME %20.12f %20.12f %20.12f' % (curtime, corr_func.real, corr_func.imag))

print('Time-propagation took %.2f seconds.\n' % (time.time() - propagation_tstart))
print('Computation finished!\n')
