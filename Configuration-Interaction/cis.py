"""
A reference implementation of Configuration Interaction Singles.
References:
"""

__authors__   = ["Boyi Zhang", "Adam S. Abbott"]
__credits__   = ["Boyi Zhang", "Adam S. Abbott", "Justin M. Turney"]

__copyright__ = "(c) 2014-2017, The Psi4NumPy Developers"
__license__   = "BSD-3-Clause"
__date__      = "2017-08-08"

# ==> Import Psi4 & NumPy <==
import psi4
import numpy as np

# ==> Set Basic Psi4 Options <==

#memory specifications
psi4.set_memory(int(2e9))
numpy_memory = 2

#output options
psi4.core.set_output_file('output.dat', False)

mol = psi4.geometry("""
0 1
O
H 1 1.1
H 1 1.1 2 104
symmetry c1
""")

psi4.set_options({'basis':        'sto-3g',
                  'scf_type':     'pk',
                  'reference':    'rhf',
                  'mp2_type':     'conv',
                  'e_convergence': 1e-8,
                  'd_convergence': 1e-8})

# Get the SCF wavefunction & energies
scf_e, scf_wfn = psi4.energy('scf', return_wfn = True)

# Check memory requirements
nmo = scf_wfn.nmo()
I_size = (nmo**4) * 8e-9
print('\nSize of the ERI tensor will be %4.2f GB.\n' % I_size)
memory_footprint = I_size * 1.5
if I_size > numpy_memory:
    psi4.core.clean()
    raise Exception("Estimated memory utilization (%4.2f GB) exceeds allotted \
                     memory limit of %4.2f GB." % (memory_footprint, numpy_memory))

# Get basis and orbital information
mints = psi4.core.MintsHelper(scf_wfn.basisset())
nbf = mints.nbf()
nalpha = scf_wfn.nalpha()
nbeta = scf_wfn.nbeta()
nocc = nalpha + nbeta
nvirt = 2 * nbf - nocc
nso = 2 * nbf 

# ==> two-electron repulsion integral <==

def spin_block_tei(I):
'''
Spin blocks two electron integrals
Using np.kron, we project I and I tranpose into the space of the 2x2 identity
The result is our two electron integral tensor in the spin orbital basis
'''
    identity = np.eye(2)
    I = np.kron(identity, I)
    return np.kron(identity, I.T)

I = np.asarray(mints.ao_eri())
I_spinblock = spin_block_tei(I)

# Converts chemists notation to physcists notation, and antisymmetrize
# (pq | rs) ---> <pr | qs>
# <pr||qs> = <pr | qs> - <pr | sq>
gao = I_spinblock.transpose(0, 2, 1, 3) - I_spinblock.transpose(0, 2, 3, 1)

# Get orbital energies, cast into NumPy array, and extend eigenvalues 
eps_a = np.asarray(scf_wfn.epsilon_a())
eps_b = np.asarray(scf_wfn.epsilon_b())
eps = np.append(eps_a, eps_b)

# Get cofficients, block, and sort 
Ca = np.asarray(scf_wfn.Ca()) 
Cb = np.asarray(scf_wfn.Cb()) 
C = np.block([ 
             [      Ca           ,   np.zeros_like(Cb) ], 
             [np.zeros_like(Ca)  ,          Cb         ] 
            ]) 
 
# Sort the columns of C according to the order of orbital energies 
C = C[:, eps.argsort()] 

# Sort orbital energies
eps = np.sort(eps) 
 
# Transform gao, which is the spin-blocked 4d array of physicist's notation, 
# antisymmetric two-electron integrals, into the MO basis using MO coefficients 
gmo = np.einsum('pQRS, pP -> PQRS',  
      np.einsum('pqRS, qQ -> pQRS',  
      np.einsum('pqrS, rR -> pqRS',  
      np.einsum('pqrs, sS -> pqrS', gao, C), C), C), C)

# Initialize CIS matrix. The dimensions are the number of possible single excitations
HCIS = np.zeros((nvirt * nocc, nvirt * nocc))

# Build the possible excitations, collect indices into a list
excitations = []
for i in range(nocc):
    for a in range(nocc, nso):
        excitations.append((i,a))

# Form matrix elements of shifted CIS Hamiltonian
for p, left_excitation in enumerate(excitations):
    i, a = left_excitation
    for q, right_excitation in enumerate(excitations):
        j, b = right_excitation
        HCIS[p, q] = (eps[a] - eps[i]) * (i == j) * (a == b) + gmo[a, j, i, b]

# Diagonalize the shifted CIS Hamiltonian
ECIS, CCIS = np.linalg.eigh(HCIS)

# Percentage contributions of coefficients for each state vector
percent_contrib = np.round(CCIS**2 * 100)


# Print detailed information on significant excitations
print('CIS:')
for state in range(len(ECIS)):
    # Print state, energy
    print('State %3d Energy (Eh) %10.7f' % (state + 1, ECIS[state]) , end = ' ')
    for idx, excitation in enumerate(excitations):
        if percent_contrib[idx, state] > 10:  
            i, a = excitation
            # Print percentage contribution and the excitation
            print('%4d%% %2d -> %2d' % (percent_contrib[idx, state], i, a), end = ' ')
    print() 

