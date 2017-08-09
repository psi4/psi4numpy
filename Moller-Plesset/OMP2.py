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

# ==> core Hamiltoniam <==

h = np.asarray(scf_wfn.H())

# Using np.kron, we project h into the space of the 2x2 identity
# The result is the core Hamiltonian in the spin orbital basis
hao = np.kron(np.eye(2), h)

# Get cofficients, block, and sort
Ca = np.asarray(scf_wfn.Ca())
Cb = np.asarray(scf_wfn.Cb())
C = np.block([
             [      Ca,         np.zeros_like(Cb)],
             [np.zeros_like(Ca),          Cb     ]])

# Get orbital energies, cast into NumPy array, and extend eigenvalues
eps_a = np.asarray(scf_wfn.epsilon_a())
eps_b = np.asarray(scf_wfn.epsilon_b())
eps = np.append(eps_a, eps_b)
# Sort the columns of C according to the order of orbital energies
C = C[:, eps.argsort()]

# ==> AO to MO transformation functions <==


def ao_to_mo(hao, C):
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
           np.einsum('pqrs, sS -> pqrS', gao, C), C), C), C)

# Transform gao and hao into MO basis
hmo = ao_to_mo(hao, C)
gmo = ao_to_mo_tei(gao, C)

# Make slices
o = slice(None, nocc)
v = slice(nocc, None)
x = np.newaxis

# Intialize t amplitude and energy
t_amp = np.zeros((nvirt, nvirt, nocc, nocc))
E_OMP2_old = 0.0

# Initialize the correlation one particle density matrix
opdm_corr = np.zeros((nso, nso))

# Build the reference one particle density matrix
opdm_ref = np.zeros((nso, nso))
opdm_ref[o, o] = np.identity(nocc)

# Initialize two particle density matrix
tpdm_corr = np.zeros((nso, nso, nso, nso))

# Initialize the rotation matrix parameter
X = np.zeros((nso, nso))
for iteration in range(MAXITER):

    # Build the Fock matrix
    f = hmo + np.einsum('piqi -> pq', gmo[:, o, :, o])

    # Build off-diagonal Fock Matrix and orbital energies
    fprime = f.copy()
    np.fill_diagonal(fprime, 0)
    eps = f.diagonal()

    # Update t amplitudes
    t1 = gmo[v, v, o, o]
    t2 = np.einsum('ac,cbij -> abij', fprime[v, v], t_amp)
    t3 = np.einsum('ki,abkj -> abij', fprime[o, o], t_amp)
    t_amp = t1 + t2 - t2.transpose((1, 0, 2, 3)) \
            - t3 + t3.transpose((0, 1, 3, 2))

    # Divide by a 4D tensor of orbital energies
    t_amp /= (- eps[v, x, x, x] - eps[x, v, x, x] +
              eps[x, x, o, x] + eps[x, x, x, o])

    # Build one particle density matrix
    opdm_corr[v, v] = (1/2)*np.einsum('ijac,bcij -> ba', t_amp.T, t_amp)
    opdm_corr[o, o] = -(1/2)*np.einsum('jkab,abik -> ji', t_amp.T, t_amp)
    opdm = opdm_corr + opdm_ref

    # Build two particle density matrix
    tpdm_corr[v, v, o, o] = t_amp
    tpdm_corr[o, o, v, v] = t_amp.T
    tpdm2 = np.einsum('rp,sq -> rspq', opdm_corr, opdm_ref)
    tpdm3 = np.einsum('rp,sq->rspq', opdm_ref, opdm_ref)
    tpdm = tpdm_corr \
        + tpdm2 - tpdm2.transpose((1, 0, 2, 3)) \
        - tpdm2.transpose((0, 1, 3, 2)) + tpdm2.transpose((1, 0, 3, 2)) \
        + tpdm3 - tpdm3.transpose((1, 0, 2, 3))

    # Newton-Raphson step
    F = np.einsum('pr,rq->pq', hmo, opdm) + (1/2) * np.einsum('prst,stqr -> pq', gmo, tpdm)
    X[v, o] = ((F - F.T)[v, o])/(- eps[v, x] + eps[x, o])

    # Build Newton-Raphson orbital rotation matrix
    U = la.expm(X - X.T)

    # Rotate spin-orbital coefficients
    C = C.dot(U)

    # Transform one and two electron integrals using new C
    hmo = ao_to_mo(hao, C)
    gmo = ao_to_mo_tei(gao, C)

    # Compute the energy
    E_OMP2 = E_nuc + np.einsum('pq,qp ->', hmo, opdm) + \
             (1/4)*np.einsum('pqrs,rspq ->', gmo, tpdm)
    print('OMP2 iteration: %3d Energy: %15.8f dE: %2.5E' % (iteration, E_OMP2, (E_OMP2-E_OMP2_old)))

    if (abs(E_OMP2-E_OMP2_old)) < E_conv:
        break

    # Updating values
    E_OMP2_old = E_OMP2


# Print final OMP2 energy
print('Final OMP2 energy: %10.8f' % (E_OMP2))
# Compare to Psi4
print('Psi4 OMP2 energy:  %10.8f' % psi4.energy('omp2'))
