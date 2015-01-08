# A simple Psi 4 input script to compute MP2 from a RHF reference
# Requirements scipy 0.13.0+ and numpy 1.7.2+
#
# Algorithms were taken directly from Daniel Crawford's programming website:
# http://sirius.chem.vt.edu/wiki/doku.php?id=crawdad:programming
# Special thanks to Rob Parrish for initial assistance with libmints
#
# Created by: Daniel G. A. Smith
# Date: 7/29/14
# License: GPL v3.0
#

import time
import numpy as np
from scipy import linalg as SLA
np.set_printoptions(precision=5, linewidth=200, suppress=True)

# Memory for Psi4 in GB
memory 2 GB

# Memory for numpy in GB
numpy_memory = 2


molecule mol {
O
H 1 1.1
H 1 1.1 2 104
symmetry c1
}


set {
basis aug-cc-pVDZ
scf_type pk
mp2_type conv
e_convergence 1e-8
d_convergence 1e-8
}

print '\nStarting RHF and integral build...'
t = time.time()

# First compute RHF energy using Psi4
energy('RHF')

# Grab data from wavfunction class 
wfn = wavefunction()
# Coefficient Matrix
C = np.array(wfn.Ca())
# Double occupied orbitals
ndocc = wfn.doccpi()[0]
# Number of molecular orbitals
nmo = wfn.nmo()
# SCF energy
SCF_E = wfn.energy()
# Orbital energies
eps = np.array(wfn.epsilon_a())

# Compute size of ERI tensor in GB
ERI_Size = (nmo**4)*8.0 / 1E9
print "Size of the ERI tensor will be %4.2f GB." % ERI_Size
memory_footprint = ERI_Size*2.5
if memory_footprint > numpy_memory:
    clean()
    raise Exception("Estimated memory utilization (%4.2f GB) exceeds numpy_memory limit of %4.2f GB." % (memory_footprint, numpy_memory))

# Integral generation from Psi4's MintsHelper
mints = MintsHelper()
I = np.array(mints.ao_eri())
I = I.reshape(nmo, nmo, nmo, nmo)


print '..total time taken for RHF and ERI integrals: %.3f seconds.' % (time.time()-t)

t=time.time()

# Split eigenvectors and eigenvalues into o and v
Co = C[:, :ndocc]
Cv = C[:, ndocc:]

# Complete the AOpqrs -> MOiajb step
MO = np.einsum('rJ,pqrs->pqJs', Co, I)
MO = np.einsum('pI,pqJs->IqJs', Co, MO)
MO = np.einsum('sB,IqJs->IqJB', Cv, MO)
MO = np.einsum('qA,IqJB->IAJB', Cv, MO)

print '\nTotal time taken for integral transformation: %.f seconds' % (time.time()-t)
print 'Shape of MO integrals: %s \n' % str(MO.shape)

# MO = np.array(NumpyHelper().mo_eri())
# MO = MO[:ndocc, ndocc:, :ndocc, ndocc:]
MO = np.array(NumpyHelper().mo_eri_subset('ACTIVE_OCC', 'ACTIVE_VIR', 'ACTIVE_OCC', 'ACTIVE_VIR'))
print MO.shape

Eocc = eps[:ndocc]
Evirt = eps[ndocc:]
e_denom = 1/(Eocc.reshape(-1, 1, 1, 1) - Evirt.reshape(-1, 1, 1) + Eocc.reshape(-1, 1) - Evirt)

# Exactly the same as MP.dat, just written in a different way
MP2corr_OS = np.einsum('iajb,iajb,iajb->', MO, MO, e_denom)
MP2corr_SS = np.einsum('iajb,iajb,iajb->', MO - MO.swapaxes(1,3), MO, e_denom)

MP2corr_E = MP2corr_SS + MP2corr_OS
MP2_E = SCF_E + MP2corr_E

SCS_MP2corr_E = MP2corr_SS/3 + MP2corr_OS*6/5
SCS_MP2_E = SCF_E + SCS_MP2corr_E

energy('MP2')
print 'MP2 SS correlation energy:         %16.10f' % MP2corr_SS
print 'MP2 OS correlation energy:         %16.10f' % MP2corr_OS

print '\nMP2 correlation energy:            %16.10f' % MP2corr_E
print 'MP2 total energy:                  %16.10f' % MP2_E
compare_values(get_variable('MP2 TOTAL ENERGY'), MP2_E, 6, 'MP2 Energy')

print '\nSCS-MP2 correlation energy:        %16.10f' % MP2corr_SS
print 'SCS-MP2 total energy:              %16.10f' % SCS_MP2_E
compare_values(get_variable('SCS-MP2 TOTAL ENERGY'), SCS_MP2_E, 6, 'SCS-MP2 Energy')


