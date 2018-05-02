"""
Reference implementation of MP3 with focus on auto-generating einsum expressions

References:
Equations & Goldstone diagram rules from [Szabo:1996]
"""

__authors__   = "Daniel G. A. Smith"
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

# Grab data from 
C = wfn.Ca()
ndocc = wfn.doccpi()[0]
nmo = wfn.nmo()
SCF_E = wfn.energy()
eps = np.asarray(wfn.epsilon_a())

# Compute size of ERI tensor in GB
ERI_Size = (nmo ** 4) * 8e-9
print('Size of the ERI/MO tensor will be %4.2f GB.' % ERI_Size)
memory_footprint = ERI_Size * 2.5
if memory_footprint > numpy_memory:
    clean()
    raise Exception("Estimated memory utilization (%4.2f GB) exceeds numpy_memory \
                    limit of %4.2f GB." % (memory_footprint, numpy_memory))

#Make spin-orbital MO
t=time.time()
print('Starting ERI build and AO -> MO transformation...')
mints = psi4.core.MintsHelper(wfn.basisset())
MO = np.asarray(mints.mo_eri(C, C, C, C))

# (pq|rs) -> <pr|qs>
MO = MO.swapaxes(1, 2)

print('..finished transformation in %.3f seconds.\n' % (time.time() - t))
print('Shape of MO integrals %s \n' % str(MO.shape))

# Build epsilon tensor
eocc = eps[:ndocc]
evirt = eps[ndocc:]
epsilon = 1/(eocc.reshape(-1, 1, 1, 1) + eocc.reshape(-1, 1, 1) - evirt.reshape(-1, 1) - evirt)

def MP_term(n, h, l, factor, string):
    """Computes value of given Goldstone diagram from MBPT expansion.

    Computes the contribution of a given Goldstone diagram to 
    the nth-order Moller--Plesset correlation energy, according
    to the rules on pp. 363 of [Szabo:1996].

    Parameters
    ----------
    n : int
        MPn order of theory
    h : int
        Number of holes
    l : int
        Number of loops
    factor : int or float
        Symmetry considerations
    string : str
        Summation string MO tensor, then energy denominators

    Returns
    -------
    term : float
        Contribution to MPn correlation energy for given Goldstone diagram.
    """   
    def get_slice(char):
        """Returns slice object for MPn expansion.

        Occupied and/or virtual orbital slices are returned according to index
        convention in [Szabo:1996]:

            Occupied indices: abcdefgh
            Virtual indicies: rstuv

        Parameters
        ----------
        char : str
            Orbital index belonging to desired slice.

        Returns
        -------
        slice 
            Slice object corresponding to orbital subset (occupied or virtual)
            to which the provided orbital index belongs.
        """
        if char in 'abcdefgh':
            return slice(0, ndocc)
        else:
            return slice(ndocc, MO.shape[0])
 
    # Compute prefactor
    pref = (-1)**(h+l) * (2**l) * float(factor)

    # Get slices
    slices = string.split(',')
    if len(slices)!=(n*2-1):
        clean()
        raise Exception('Number of terms does not match the order of pertubation theory')

    # Create views
    views = []
    
    # MO views
    for term in range(n):
        tmp_slice = slices[term]
        tmp_slice = [get_slice(x) for x in tmp_slice]
        views.append(MO[tmp_slice[0], tmp_slice[1], tmp_slice[2], tmp_slice[3]])
    
    # Epsilon views 
    for term in range(n-1):
        views.append(epsilon)
    
    # Compute term!
    string += '->'
    term = np.einsum(string, *views)
    term *= pref
    return term

### MP2

MP2corr_E = MP_term(2, 2, 2, 0.5, 'abrs,rsab,abrs')
MP2corr_E += MP_term(2, 2, 1, 0.5, 'abrs,rsba,abrs')
MP2total_E = SCF_E + MP2corr_E
print('MP2 correlation energy: %.8f' % MP2corr_E)
print('MP2 total energy:       %.8f' % MP2total_E)
psi4.compare_values(psi4.energy('MP2'), MP2total_E, 6, 'MP2 Energy')

### MP3 Correlation Energy
### Terms taken from [Szabo:1996] Tbl. 6.2, pp. 364-365

print('\nStarting MP3 correlation energy...')
# MP3 Eqn 1: 3rd order diagram 1
MP3corr_E =  MP_term(3, 2, 2, 0.5, 'abru,ruts,tsab,abru,abts') 
# MP3 Eqn 2: 3rd order diagram 2
MP3corr_E += MP_term(3, 4, 2, 0.5, 'adrs,cbad,rscb,adrs,cbrs')
# MP3 Eqn 3: 3rd order diagram 3
MP3corr_E += MP_term(3, 3, 2, 1.0, 'acrt,rbsc,stab,acrt,abst')
# MP3 Eqn 4: 3rd order diagram 4 
MP3corr_E += MP_term(3, 3, 2, 1.0, 'bcrt,rasb,stac,bcrt,acst')
# MP3 Eqn 5: 3rd order diagram 5 
MP3corr_E += MP_term(3, 3, 3, 1.0, 'acrt,btsc,rsab,acrt,abrs')
# MP3 Eqn 6: 3rd order diagram 6 
MP3corr_E += MP_term(3, 3, 1, 1.0, 'cbrt,atsc,rsab,cbrt,abrs')
# MP3 Eqn 7: 3rd order diagram 7 
MP3corr_E += MP_term(3, 4, 1, 0.5, 'acrs,dbac,srdb,acrs,dbrs')
# MP3 Eqn 8: 3rd order diagram 8 
MP3corr_E += MP_term(3, 2, 1, 0.5, 'abrt,trus,usab,abtr,abus')
# MP3 Eqn 9: 3rd order diagram 9
MP3corr_E += MP_term(3, 3, 1, 1.0, 'bcrt,arbs,tsac,cbrt,acst')
# MP3 Eqn 10: 3rd order diagram 10
MP3corr_E += MP_term(3, 3, 1, 1.0, 'cbrt,rasb,stac,cbrt,acst')
# MP3 Eqn 11: 3rd order diagram 11
MP3corr_E += MP_term(3, 3, 2, 1.0, 'abrs,scat,rtbc,abrs,cbrt')
# MP3 Eqn 12: 3rd order diagram 12
MP3corr_E += MP_term(3, 3, 2, 1.0, 'bcrt,atsc,rsab,bctr,abrs')

print('...took %.3f seconds to compute MP3 correlation energy.\n' % (time.time() - t))

print('Third order energy:     %.8f' % MP3corr_E)
MP3corr_E += MP2corr_E
MP3total_E = SCF_E + MP3corr_E
print('MP3 correlation energy: %.8f' % MP3corr_E)
print('MP3 total energy:       %.8f' % MP3total_E)

# Compare to Psi4
psi4.compare_values(psi4.energy('MP3'), MP3total_E, 6, 'MP3 Energy')


