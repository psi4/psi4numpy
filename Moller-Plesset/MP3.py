"""
Reference implementation for the correlation energy of MP3 with an RHF reference.

References:
- Equations from [Szabo:1996]
"""

__authors__    = "Daniel G. A. Smith"
__credits__   = ["Daniel G. A. Smith", "Dominic A. Sirianni"]

__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__   = "BSD-3-Clause"
__date__      = "2017-05-23"

import time
import numpy as np
np.set_printoptions(precision=5, linewidth=200, suppress=True)
import psi4

# Memory for Psi4 in GB
psi4.set_memory('2 GB')
psi4.core.set_output_file('output.dat', False)

# Memory for numpy in GB
numpy_memory = 2

mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
symmetry c1
""")

psi4.set_options({'basis': 'aug-cc-pvdz',
                  'scf_type': 'pk',
                  'guess': 'core',
                  'mp2_type': 'conv',
                  'freeze_core': 'false',
                  'e_convergence': 1e-8,
                  'd_convergence': 1e-8})

# First compute RHF energy using Psi4
scf_e, wfn = psi4.energy('SCF', return_wfn=True)

# Coefficient Matrix
C = np.array(wfn.Ca())
# Double occupied orbitals
ndocc = wfn.doccpi()[0]
# Number of molecular orbitals
nmo = wfn.nmo()
# SCF energy
SCF_E = wfn.energy()
# Orbital energies
eps = wfn.epsilon_a()
eps = np.array([eps.get(x) for x in range(C.shape[0])])

# Compute size of ERI tensor in GB
ERI_Size = (nmo**4)*8.0 / 1E9
print("Size of the ERI tensor will be %4.2f GB." % ERI_Size)
memory_footprint = ERI_Size*2.5
if memory_footprint > numpy_memory:
    clean()
    raise Exception("Estimated memory utilization (%4.2f GB) exceeds numpy_memory limit of %4.2f GB." % (memory_footprint, numpy_memory))

# Integral generation from Psi4's MintsHelper
t = time.time()
mints = psi4.core.MintsHelper(wfn.basisset())
I = np.array(mints.ao_eri())
I = I.reshape(nmo, nmo, nmo, nmo)

print('\nTotal time taken for ERI integrals: %.3f seconds.' % (time.time()-t))

t=time.time()

# Complete the AOpqrs -> MOiajb step
MO = np.einsum('rJ,pqrs->pqJs', C, I)
MO = np.einsum('pI,pqJs->IqJs', C, MO)
MO = np.einsum('sB,IqJs->IqJB', C, MO)
MO = np.einsum('qA,IqJB->IAJB', C, MO)

# (pq|rs) -> <ps|rq>
MO = MO.swapaxes(1, 2)

print('\nTotal time taken for integral transformation: %.f seconds' % (time.time()-t))
print('Shape of MO integrals %s \n' % str(MO.shape))

# Build epsilon tensor
eocc = eps[:ndocc]
evirt = eps[ndocc:]
epsilon = 1/(eocc.reshape(-1, 1, 1, 1) + eocc.reshape(-1, 1, 1) - evirt.reshape(-1, 1) - evirt)

# Build o and v slices
o = slice(0, ndocc)
v = slice(ndocc, MO.shape[0])

### MP2 correlation energy

MP2corr_E = 2 * np.einsum('abrs,rsab,abrs', MO[o, o, v, v], MO[v, v, o, o], epsilon)
MP2corr_E -= np.einsum('abrs,rsba,abrs', MO[o, o, v, v], MO[v, v, o, o], epsilon)
MP2total_E = SCF_E + MP2corr_E
print('MP2 correlation energy: %16.8f' % MP2corr_E)
print('MP2 total energy:       %16.8f' % MP2total_E)
psi4.compare_values(psi4.energy('MP2'), MP2total_E, 6, 'MP2 Energy')

print('\n Starting MP3 energy...')
t = time.time()

# MP3 Correlation energy

# Prefactors taken from terms in unnumbered expression for spatial-orbital MP3
# energy on [Szabo:1996] pp. (bottom) 367 - (top) 368. Individual equations taken
# from [Szabo:1996] Tbl. 6.2 pp. 364-365

# Equation 1: 3rd order diagram 1
MP3corr_E =   2.0 * np.einsum('abru,ruts,tsab,abru,abts', MO[o, o, v, v], MO[v, v, v, v], MO[v, v, o, o], epsilon, epsilon) 
# Equation 2: 3rd order diagram 2 
MP3corr_E +=  2.0 * np.einsum('adrs,cbad,rscb,adrs,cbrs', MO[o, o, v, v], MO[o, o, o, o], MO[v, v, o, o], epsilon, epsilon)
# Equation 3: 3rd order diagram 3
MP3corr_E += -4.0 * np.einsum('acrt,rbsc,stab,acrt,abst', MO[o, o, v, v], MO[v, o, v, o], MO[v, v, o, o], epsilon, epsilon)
# Equation 4: 3rd order diagram 4
MP3corr_E += -4.0 * np.einsum('bcrt,rasb,stac,bcrt,acst', MO[o, o, v, v], MO[v, o, v, o], MO[v, v, o, o], epsilon, epsilon)
# Equation 5: 3rd order diagram 5
MP3corr_E +=  8.0 * np.einsum('acrt,btsc,rsab,acrt,abrs', MO[o, o, v, v], MO[o, v, v, o], MO[v, v, o, o], epsilon, epsilon)
# Equation 6: 3rd order diagram 6
MP3corr_E +=  2.0 * np.einsum('cbrt,atsc,rsab,cbrt,abrs', MO[o, o, v, v], MO[o, v, v, o], MO[v, v, o, o], epsilon, epsilon)
# Equation 7: 3rd order diagram 7
MP3corr_E += -1.0 * np.einsum('acrs,dbac,srdb,acrs,dbrs', MO[o, o, v, v], MO[o, o, o, o], MO[v, v, o, o], epsilon, epsilon)
# Equation 8: 3rd order diagram 8
MP3corr_E += -1.0 * np.einsum('abrt,trus,usab,abtr,abus', MO[o, o, v, v], MO[v, v, v, v], MO[v, v, o, o], epsilon, epsilon)
# Equation 9: 3rd order diagram 9
MP3corr_E +=  2.0 * np.einsum('bcrt,arbs,tsac,cbrt,acst', MO[o, o, v, v], MO[o, v, o, v], MO[v, v, o, o], epsilon, epsilon)
# Equation 10: 3rd order diagram 10
MP3corr_E +=  2.0 * np.einsum('cbrt,rasb,stac,cbrt,acst', MO[o, o, v, v], MO[v, o, v, o], MO[v, v, o, o], epsilon, epsilon)
# Equation 11: 3rd order diagram 11
MP3corr_E += -4.0 * np.einsum('abrs,scat,rtbc,abrs,cbrt', MO[o, o, v, v], MO[v, o, o, v], MO[v, v, o, o], epsilon, epsilon)
# Equation 12: 3rd order diagram 12
MP3corr_E += -4.0 * np.einsum('bcrt,atsc,rsab,bctr,abrs', MO[o, o, v, v], MO[o, v, v, o], MO[v, v, o, o], epsilon, epsilon)

print('...took %.3f seconds to compute MP3 correlation energy.\n' % (time.time()-t))

print('Third order energy:     %16.8f' % MP3corr_E)
MP3corr_E += MP2corr_E
MP3total_E = SCF_E + MP3corr_E
print('MP3 correlation energy: %16.8f' % MP3corr_E)
print('MP3 total energy:       %16.8f' % MP3total_E)
psi4.compare_values(psi4.energy('MP3'), MP3total_E, 6, 'MP3 Energy')


