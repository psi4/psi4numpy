"""
Unrestricted open-shell Hartree-Fock using direct second-order
convergence acceleration.

References:
- UHF equations & algorithm from [Szabo:1996]
- SO equations & algorithm from [Helgaker:2000]
"""

__authors__ = "Daniel G. A. Smith"
__credits__ = ["Daniel G. A. Smith"]

__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__ = "BSD-3-Clause"
__date__ = "2017-9-30"

import time
import numpy as np
np.set_printoptions(precision=5, linewidth=200, suppress=True)
import psi4

# Set Psi4 memory and output options
psi4.set_memory('2 GB')
psi4.core.set_output_file('output.dat', False)

# Triplet oxygen
mol = psi4.geometry("""
    0 3
    O
    O 1 1.2
symmetry c1
""")

psi4.set_options({'basis': 'aug-cc-pvdz',
                  'reference': 'uhf'})

# Set defaults
maxiter = 10
E_conv = 1.0E-13
D_conv = 1.0E-13

# Integral generation from Psi4's MintsHelper
wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option('BASIS'))
mints = psi4.core.MintsHelper(wfn.basisset())
S = np.asarray(mints.ao_overlap())
V = np.asarray(mints.ao_potential())
T = np.asarray(mints.ao_kinetic())

# Occupations
nbf = wfn.nso()
nalpha = wfn.nalpha()
nbeta = wfn.nbeta()

if nbf > 100:
    raise Exception("This has a N^4 memory overhead, killing if nbf > 100.")

H = T + V

# Orthogonalizer A = S^(-1/2)
A = mints.ao_overlap()
A.power(-0.5, 1.e-16)
A = np.asarray(A)

# ERI's
I = np.asarray(mints.ao_eri())

# Steal a good starting guess
psi4.set_options({'e_convergence': 1e-4,
                  'd_convergence': 1e-4,
                  'maxiter': 7,
                  'guess': 'sad'})

scf_e, wfn = psi4.energy('SCF', return_wfn=True)
Ca = np.array(wfn.Ca())
Da = np.array(wfn.Da())
Cb = np.array(wfn.Cb())
Db = np.array(wfn.Db())

nalpha = wfn.nalpha()
nbeta = wfn.nbeta()

t = time.time()
E = 0.0
Enuc = mol.nuclear_repulsion_energy()
Eold = 0.0

print('\nTotal time taken for setup: %.3f seconds' % (time.time() - t))

print('\nStart SCF iterations:\n')
t = time.time()


def transform(I, C1, C2, C3, C4):
    #MO = np.einsum('pA,pqrs->Aqrs', C1,  I)
    nao = I.shape[0]
    MO = np.dot(C1.T, I.reshape(nao, -1)).reshape(C1.shape[1], nao, nao, nao)

    MO = np.einsum('qB,Aqrs->ABrs', C2, MO)
    MO = np.einsum('rC,ABrs->ABCs', C3, MO)
    MO = np.einsum('sD,ABCs->ABCD', C4, MO)
    return MO

# Rotate orbs and produce C and D matrices
def rotate_orbs(C, x, nocc):
    U = np.zeros_like(C)
    U[:nocc, nocc:] = x
    U[nocc:, :nocc] = -x.T

    expU = U.copy()
    expU[np.diag_indices_from(U)] += 1
    expU += 0.5 * np.dot(U, U)

    expU, r = np.linalg.qr(expU.T)
    Cn = C.dot(expU)
    D = np.dot(Cn[:,:nocc], Cn[:,:nocc].T)
    return (Cn, D)

for SCF_ITER in range(1, maxiter + 1):

    # Build the alpha & beta Fock matrices
    Ja = np.einsum('pqrs,rs->pq', I, Da)
    Ka = np.einsum('prqs,rs->pq', I, Da)
    Jb = np.einsum('pqrs,rs->pq', I, Db)
    Kb = np.einsum('prqs,rs->pq', I, Db)

    Fa = H + (Ja + Jb) - Ka
    Fb = H + (Ja + Jb) - Kb

    # dRMS error
    diisa_e = A.dot(Fa.dot(Da).dot(S) - S.dot(Da).dot(Fa)).dot(A)
    diisb_e = A.dot(Fb.dot(Db).dot(S) - S.dot(Db).dot(Fb)).dot(A)

    # SCF energy and update: [Szabo:1996], exercise 3.40, pp. 215
    SCF_E = np.einsum('pq,pq->', Da + Db, H)
    SCF_E += np.einsum('pq,pq->', Da, Fa)
    SCF_E += np.einsum('pq,pq->', Db, Fb)
    SCF_E *= 0.5
    SCF_E += Enuc

    dRMS = 0.5 * (np.mean(diisa_e**2)**0.5 + np.mean(diisb_e**2)**0.5)
    print('SCF Iteration %3d: Energy = %4.16f   dE = % 1.5E   dRMS = %1.5E'
          % (SCF_ITER, SCF_E, (SCF_E - Eold), dRMS))
    if (abs(SCF_E - Eold) < E_conv) and (dRMS < D_conv):
        break

    Eold = SCF_E

    Cocca = Ca[:, :nalpha]
    Cvira = Ca[:, nalpha:]

    Coccb = Cb[:, :nbeta]
    Cvirb = Cb[:, nbeta:]

    # Form gradients from MO Fock matrices: [Helgaker:2000] Eqn. 10.8.34, pp. 484
    moFa = (Ca.T).dot(Fa).dot(Ca)
    moFb = (Cb.T).dot(Fb).dot(Cb)
    grada = -4 * moFa[:nalpha, nalpha:]
    gradb = -4 * moFb[:nbeta, nbeta:]

    # Form off diagonal contributions to Hessian
    Jab = 8 * transform(I, Cocca, Cvira, Coccb, Cvirb)

    # Form diagonal alpha contributions
    MOaa = transform(I, Cocca, Ca, Ca, Ca)
    Ha  = np.einsum('ab,ij->iajb', moFa[nalpha:, nalpha:], np.diag(np.ones(nalpha)))
    Ha -= np.einsum('ij,ab->iajb', moFa[:nalpha:, :nalpha], np.diag(np.ones(nbf-nalpha)))
    Ha += 2 * MOaa[:, nalpha:, :nalpha, nalpha:]
    Ha -= MOaa[:, nalpha:, :nalpha, nalpha:].swapaxes(0, 2)
    Ha -= MOaa[:, :nalpha, nalpha:, nalpha:].swapaxes(1, 2)
    Ha *= 4

    # Form diagonal beta contributions
    MObb = transform(I, Coccb, Cb, Cb, Cb)
    Hb  = np.einsum('ab,ij->iajb', moFb[nbeta:, nbeta:], np.diag(np.ones(nbeta)))
    Hb -= np.einsum('ij,ab->iajb', moFb[:nbeta:, :nbeta], np.diag(np.ones(nbf-nbeta)))
    Hb += 2 * MObb[:, nbeta:, :nbeta, nbeta:]
    Hb -= MObb[:, nbeta:, :nbeta, nbeta:].swapaxes(0, 2)
    Hb -= MObb[:, :nbeta, nbeta:, nbeta:].swapaxes(1, 2)
    Hb *= 4

    # Build the full Hessian matrix
    na = Ha.shape[0] * Ha.shape[1]
    nb = Hb.shape[0] * Hb.shape[1]
    ntot = na + nb

    #  aa | ab
    #  -------
    #  ba | bb
    Hess = np.zeros((ntot, ntot))
    Hess[:na,:na] = Ha.reshape(na, na)
    Hess[:na,na:] = Jab.reshape(na,nb)
    Hess[na:,:na] = Jab.reshape(na,nb).T
    Hess[na:,na:] = Hb.reshape(nb, nb)

    # Invert hessian and obtain new vectors
    Hinv = np.linalg.inv(Hess)

    gradvec = np.hstack((grada.reshape(-1), gradb.reshape(-1)))
    resultx = np.einsum('ij,j->i', Hinv, gradvec)

    xa = resultx[:na].reshape(Ha.shape[0], Ha.shape[1])
    xb = resultx[na:].reshape(Hb.shape[0], Hb.shape[1])

    # Rotate the orbitals
    Ca, Da = rotate_orbs(Ca, xa, nalpha)
    Cb, Db = rotate_orbs(Cb, xb, nbeta)

    if SCF_ITER == maxiter:
        clean()
        raise Exception("Maximum number of SCF cycles exceeded.")

print('Total time for SCF iterations: %.3f seconds \n' % (time.time() - t))

spin_mat = (Cb[:, :nbeta].T).dot(S).dot(Ca[:, :nalpha])
spin_contam = min(nalpha, nbeta) - np.vdot(spin_mat, spin_mat)
print('Spin Contamination Metric: %1.5E\n' % spin_contam)

print('Final SCF energy: %.8f hartree' % SCF_E)

# Compare to Psi4
psi4.set_options({'e_convergence': 1e-8,
                  'r_convergence': 1e-8,
                  'scf_type': 'pk',
                  'maxiter': 100})

SCF_E_psi = psi4.energy('SCF')
psi4.compare_values(SCF_E_psi, SCF_E, 6, 'SCF Energy')
