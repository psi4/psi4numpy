
"""
A reference implementation of orbital optimized second-order Moller-Plesset perturbation theory.

References:
"""

__authors__    = "Boyi Zhang"
__credits__   = ["Boyi Zhang", "Justin M. Turney"]

__copyright_amp__ = "(c) 2014-2017, The Psi4NumPy Developers"
__license__   = "BSD-3-Clause"
__date__      = "2017-08-08"

# ==> Import Psi4, NumPy, & SciPy <==
import psi4
import numpy as np
import scipy.linalg as la

# ==> Set Basic Psi4 Options <==

# Memory specifications
psi4.set_memory(int(2e9))
numpy_memory = 2

# Output options
psi4.core.set_output_file('output.dat', False)

mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
symmetry c1
""")

psi4.set_options({'basis':'sto-3g',
                    'scf_type': 'pk',
                    'mp2_type' : 'conv',
                    'reference' : 'uhf',
                    'e_convergence' : 1e-8,
                    'd_convergence' : 1e-8})

# Get the SCF wavefunction & energies
scf_e, scf_wfn = psi4.energy('scf', return_wfn=True)

# ==> Nuclear Repulsion Energy <==
E_nuc = mol.nuclear_repulsion_energy()

# ==> Set default program options <==
# Maximum OMP2 iterations
MAXITER = 40
# Energy convergence criterion
E_conv = 1.0e-8

# Check memory requirements
nmo = scf_wfn.nmo()
I_size = (nmo**4) * 8e-9
print('\nSize of the ERI tensor will be %4.2f GB.\n' % I_size)
memory_footprint = I_size * 1.5
if I_size > numpy_memory:
    psi4.core.clean()
    raise Exception("Estimated memory utilization (%4.2f GB) exceeds allotted \
                     memory limit of %4.2f GB." % (memory_footprint, numpy_memory))


# Create instance of MintsHelper class
mints = psi4.core.MintsHelper(scf_wfn.basisset())

# Get basis and orbital information
nbf = mints.nbf()
nalpha = scf_wfn.nalpha()
nbeta = scf_wfn.nbeta()
nocc = nalpha + nbeta
nvirt = 2 * nbf - nocc
nso = 2 * nbf 

# ==> core Hamiltoniam <==

h = np.asarray(scf_wfn.H())

# Using np.kron, we project h into the space of the 2x2 identity
# The result is the core Hamiltonian in the spin orbital basis
hao = np.kron(np.eye(2), h)

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

# ==> AO to MO transformation functions <==
def ao_to_mo(hao,C):
    '''
    Transform hao, which is the core Hamiltonian in the spin orbital basis,
    into the MO basis using MO coefficients 
    '''

    return np.einsum('pQ, pP -> PQ',
           np.einsum('pq, qQ -> pQ', hao, C), C)        

def ao_to_mo_tei(gao, C):
    '''
    Transform gao, which is the spin-blocked 4d array of physicist's notation, 
    antisymmetric two-electron integrals, into the MO basis using MO coefficients 
    '''
    return np.einsum('pQRS, pP -> PQRS',
           np.einsum('pqRS, qQ -> pQRS',
           np.einsum('pqrS, rR -> pqRS',
           np.einsum('pqrs, sS -> pqrS', gao, C),C),C),C)

# Transform gao and hao into MO basis 
hmo_old = ao_to_mo(hao, C)
gmo_old = ao_to_mo_tei(gao, C)

# Intialize t amplitude and energy
t_amp_old = np.zeros((nocc, nocc, nvirt, nvirt))
E_OMP2_old = 0.0

# Make slices
o = slice(None, nocc)
v = slice(nocc, None)
x = np.newaxis

# Initialize one particle density matrix
odm_tilde = np.zeros((nso, nso))
odmref = np.zeros((nso, nso))
odmref[o,o] = np.identity(nocc)

# Initialize two particle density matrix
tdm_tilde = np.zeros((nso, nso, nso, nso)) 

X = np.zeros((nso, nso))        

for iteration in range(MAXITER):

    # Build the Fock matrix
    f = hmo_old + np.einsum('piqi -> pq',gmo_old[:,o,:,o])

    # Bild off-diagonal Fock Matrix and orbital energies
    fprime = f.copy()
    np.fill_diagonal(fprime, 0)
    eps = f.diagonal()

    # Update t amplitudes
    t1 = gmo_old[o,o,v,v] 
    t2 = np.einsum('ac,ijcb -> ijab',fprime[v,v],t_amp_old)
    t3 = np.einsum('ki,kjab -> ijab', fprime[o,o],t_amp_old)
    t_amp = t1 + t2 - t2.transpose((0,1,3,2)) - t3 + t3.transpose((1,0,2,3))
    # Divide by a 4D tensor of orbital energies
    t_amp /= (eps[o,x,x,x] + eps[x,o,x,x] - eps[x,x,v,x] - eps[x,x,x,v])

    #Build one particle density matrix
    odm_tilde[v,v] =  (1/2)*np.einsum('ijac,ijbc -> ab', t_amp, t_amp)            
    odm_tilde[o,o] = -(1/2)*np.einsum('jkab,ikab -> ij', t_amp, t_amp)
    odm = odm_tilde + odmref

    #Build two particle density matrix
    tdm_tilde[v,v,o,o] = t_amp.T
    tdm_tilde[o,o,v,v] = t_amp
    tdm2 = np.einsum('pr,qs -> pqrs', odm_tilde, odmref)
    tdm3 = np.einsum('pr,qs->pqrs', odmref, odmref)
    tdm = tdm_tilde \
          + tdm2 - tdm2.transpose((1,0,2,3)) - tdm2.transpose((0,1,3,2)) + tdm2.transpose((1,0,3,2)) \
          + tdm3 - tdm3.transpose((0,1,3,2))

    # Newton-Raphson step
    F = np.einsum('rp,qr->pq', hmo_old, odm)+(1/2) * np.einsum('prst,qrst -> pq', gmo_old, tdm)
    X[o,v] = ((F-F.T)[o,v])/(eps[o,x]-eps[x,v])

    # Build Newton-Raphson orbital rotation matrix
    U = la.expm(X-X.T)
    
    # Rotate spin-orbital coefficients
    C = C.dot(U) 

    # Transform one and two electron integrals using new C
    hmo = ao_to_mo(hao, C)
    gmo = ao_to_mo_tei(gao, C)

    # Compute the energy
    E_OMP2 = E_nuc + np.einsum('pq,pq ->', hmo, odm) + (1/4)*np.einsum('pqrs,pqrs ->', gmo, tdm)
    print('OMP2 iteration %3d: Energy %20.10f dE %2.5E'%(iteration, E_OMP2, (E_OMP2-E_OMP2_old)))

    if (abs(E_OMP2-E_OMP2_old)) < 1.e-10:
        break

    #updating values
    gmo_old = gmo
    hmo_old = hmo
    t_amp_old = t_amp
    E_OMP2_old = E_OMP2

print('The final OMP2 energy is {:20.10f}'.format(E_OMP2))
print(psi4.energy('omp2'))


