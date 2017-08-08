# ==> Import Psi4, NumPy, & SciPy <==
import psi4
import numpy as np
import scipy.linalg as la

# ==> Set Basic Psi4 Options <==

#memory specifications
psi4.set_memory(int(2e9))
numpy_memory = 2

#output options
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

# ==> Set default program options <==
# Maximum OMP2 iterations
MAXITER = 40
# Energy convergence criterion
E_conv = 1.0e-8

# ==> Necessary functions <==
def ao_to_mo(hao,C):

    return np.einsum('pQ, pP -> PQ',
           np.einsum('pq, qQ -> pQ', hao, C), C)        

def ao_to_mo_tei(gao, C):

    return np.einsum('pQRS, pP -> PQRS',
           np.einsum('pqRS, qQ -> pQRS',
           np.einsum('pqrS, rR -> pqRS',
           np.einsum('pqrs, sS -> pqrS', gao, C),C),C),C)

def spin_block_tei(I):
    identity = np.eye(2)
    I = np.kron(identity, I)
    return np.kron(identity, I.T)

# Get the SCF wavefunction & energies
scf_e, scf_wfn = psi4.energy('scf', return_wfn=True)

# ==> Get orbital information & energy eigenvalues <==

# Get orbital energies, cast into NumPy array, and extend eigenvalues 
eps_a = np.asarray(scf_wfn.epsilon_a())
eps_b = np.asarray(scf_wfn.epsilon_b())
eps = np.append(eps_a, eps_b)

# Number of Occupied orbitals & MOs
nalpha = scf_wfn.nalpha()
nbeta = scf_wfn.nbeta()
nmo = scf_wfn.nmo()

#update nocc and nvirt
nso = nmo * 2
nocc = nalpha + nbeta
nvirt = nso - nocc

# Get cofficients, block diagonalize, and sort 
Ca = np.asarray(scf_wfn.Ca())
Cb = np.asarray(scf_wfn.Cb())
C = np.block([
             [       Ca         , np.zeros_like(Cb)],
             [np.zeros_like(Ca) ,        Cb       ]
             ])
C = C[:, eps.argsort()]

# ==> Nuclear Repulsion Energy <==
E_nuc = mol.nuclear_repulsion_energy()

# ==> core Hamiltoniam <==

h = np.asarray(scf_wfn.H())
hao = np.kron(np.eye(2), h)

# ==> ERIs <==
# Create instance of MintsHelper class
mints = psi4.core.MintsHelper(scf_wfn.basisset())

# Memory check for ERI tensor
ERI_size = (nmo**4) * 8.e-9
print('\nSize of the ERI tensor will be %4.2f GB.' % ERI_size)
memory_footprint = ERI_size * 5.2
if memory_footprint > numpy_memory:
    psi4.core.clean()
    raise Exception("Estimated memory utilization (%4.2f GB) exceeds allotted memory \
                     limit of %4.2f GB." % (memory_footprint, numpy_memory))

#Make spin-orbital MO and extend eigenvalues
I = np.asarray(mints.ao_eri())
I_spinblock = spin_block_tei(I)

#Convert MO to physicists notation and antisymmetrize
#insert equation 
gao = I_spinblock.transpose(0,2,1,3) - I_spinblock.transpose(0,2,3,1) 

#Make slices
o = slice(None, nocc)
v = slice(nocc, None)
x = np.newaxis

odm = np.zeros((nso, nso))
tdm = np.zeros((nso, nso, nso, nso)) 
odmref = np.zeros((nso, nso))
odmref[o,o] = np.identity(nocc)
X = np.zeros((nso, nso))        

hmo_old = ao_to_mo(hao, C)
gmo_old = ao_to_mo_tei(gao, C)
t_old = np.zeros((nocc, nocc, nvirt, nvirt))
E_OMP2_old = 0.0

for iteration in range(MAXITER):
    f = hmo_old + np.einsum('piqi -> pq',gmo_old[:,o,:,o])
    #off diagonal Fock Matrix
    fprime = f.copy()
    np.fill_diagonal(fprime, 0)
    #updated orbital energies 
    eps = f.diagonal()
    # t amplitudes
    t2 = np.einsum('ac,ijcb -> ijab',fprime[v,v],t_old)
    t3 = np.einsum('ki,kjab -> ijab', fprime[o,o],t_old)
    t = (gmo_old[o,o,v,v] + t2 - t2.transpose((0,1,3,2)) - t3 + t3.transpose((1,0,2,3)))
    t /= (eps[o,x,x,x]+eps[x,o,x,x]-eps[x,x,v,x]-eps[x,x,x,v])
    #one and two particle density matrices 
    odm[v,v] = (1/2)*np.einsum('ijac,ijbc -> ab',t,t)            
    odm[o,o] = -(1/2)*np.einsum('jkab,ikab -> ij',t,t)
    tdm[v,v,o,o] = t.T
    tdm[o,o,v,v] = t
    tdm2 = np.einsum('pr,qs -> pqrs', odm,odmref)
    tdm3 = np.einsum('pr,qs->pqrs', odmref,odmref)
    odm_gen = odm + odmref
    tdm_gen = tdm + tdm2 - tdm2.transpose((1,0,2,3))-tdm2.transpose((0,1,3,2))+tdm2.transpose((1,0,3,2)) + tdm3 - tdm3.transpose((0,1,3,2))
    print(np.trace(odm_gen))
    #Newton-Raphson
    F = np.einsum('rp,qr->pq',hmo_old,odm_gen)+(1/2)*np.einsum('prst,qrst -> pq',gmo_old,tdm_gen)
    X[o,v] = ((F-F.T)[o,v])/(eps[o,x]-eps[x,v])
    #rotate coefficients
    U = la.expm(X-X.T)
    C = C.dot(U) 
    #transform integrals
    hmo = ao_to_mo(hao,C)
    gmo = ao_to_mo_tei(gao,C)
    # get energy
    E_OMP2 = E_nuc + np.einsum('pq,pq ->',hmo,odm_gen) + (1/4)*np.einsum('pqrs,pqrs ->',gmo,tdm_gen)
    print('OMP2 iteration{:3d}: energy {:20.10f} dE {:2.5E}'.format(iteration,E_OMP2,(E_OMP2-E_OMP2_old)))

    if (abs(E_OMP2-E_OMP2_old)) < 1.e-10:
        break

    #updating values
    gmo_old = gmo
    hmo_old = hmo
    t_old = t
    E_OMP2_old = E_OMP2

print('The final OMP2 energy is {:20.10f}'.format(E_OMP2))
print(psi4.energy('omp2'))

