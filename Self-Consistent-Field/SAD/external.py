import time
import numpy as np

class DIIS_helper(object):

    def __init__(self, max_vec=6):
        self.error = []
        self.vector = []
        self.max_vec = max_vec

    def add(self, matrix, error):
        if len(self.error) > 1:
            if self.error[-1].shape[0] != error.size:
                raise Exception("Error vector size does not match previous vector.")
            if self.vector[-1].shape != matrix.shape:
                raise Exception("Vector shape does not match previous vector.")

        self.error.append(error.ravel().copy())
        self.vector.append(matrix.copy())

    def extrapolate(self):
        # Limit size of DIIS vector
        diis_count = len(self.vector)

        if diis_count == 0:
            raise Exception("DIIS: No previous vectors.")
        if diis_count == 1:
            return self.vector[0]

        if diis_count > self.max_vec:
            # Remove oldest vector
            del self.vector[0]
            del self.error[0]
            diis_count -= 1

        # Build error matrix B
        B = np.empty((diis_count + 1, diis_count + 1))
        B[-1, :] = -1
        B[:, -1] = -1
        B[-1, -1] = 0
        for num1, e1 in enumerate(self.error):
            B[num1, num1] = np.vdot(e1, e1)
            for num2, e2 in enumerate(self.error):
                if num2 >= num1: continue
                val = np.vdot(e1, e2)
                B[num1, num2] = B[num2, num1] = val

        # normalize
        B[abs(B) < 1.e-14] = 1.e-14
        B[:-1, :-1] /= np.abs(B[:-1, :-1]).max()
        # Build residual vector
        resid = np.zeros(diis_count + 1)
        resid[-1] = -1

        # Solve pulay equations
        ci = np.dot(np.linalg.pinv(B), resid)

        # combination of previous fock matrices
        V = np.zeros_like(self.vector[-1])
        for num, c in enumerate(ci[:-1]):
            V += c * self.vector[num]

        return V

def uhf(psi4, mol, basis, pop_a=None, pop_b=None, guess_a=None, guess_b=None, do_print=True, E_conv = 1.e-7, D_conv = 1.e-5):

    # Set defaults
    maxiter = 40
    #E_conv = 1.0E-7
    #D_conv = 1.0E-5

    psi4.set_global_option('BASIS', basis)
    # Integral generation from Psi4's MintsHelper
    t = time.time()
    wfn = psi4.new_wavefunction(mol, psi4.get_global_option('BASIS'))
    mints = psi4.MintsHelper(wfn.basisset())
    S = np.asarray(mints.ao_overlap())

    # Get nbf and ndocc for closed shell molecules
    nbf = S.shape[0]
    if pop_a is not None:
        nocca = pop_a.shape[0]
        if pop_b is None:
            noccb = nocca
            pop_b = noccb
        else:
            noccb = pop_b.shape[0]
    else:
        nocca = wfn.nalpha()
        noccb = wfn.nbeta()

    if do_print:
        print '\nNumber of alpha orbitals:   %3d' % nocca
        print 'Number of beta orbitals:    %3d' % noccb
        print 'Number of basis functions:  %3d' % nbf

    V = np.asarray(mints.ao_potential())
    T = np.asarray(mints.ao_kinetic())

    if do_print:
        print '\nTotal time taken for integrals: %.3f seconds.' % (time.time()-t)

    t = time.time()

    # Build H_core
    H = T + V

    # Orthogonalizer A = S^(-1/2)
    A = mints.ao_overlap()
    A.power(-0.5, 1.e-16)
    A = np.asarray(A)

    def diag_H(H, nocc, pop=None):
        Hp = A.dot(H).dot(A)
        e, C2 = np.linalg.eigh(Hp)
        C = A.dot(C2)
        Cocc = C[:, :nocc]
        if pop is not None:
            Cocc *= pop
        D = np.einsum('pi,qi->pq', Cocc, Cocc)
        return (C, D)

    if guess_a is None:
        Ca, Da = diag_H(H, nocca, pop_a)
        Cb, Db = diag_H(H, noccb, pop_b)
    else:
        Ca = guess_a
        if guess_b is None:
            Cb = guess_a
        Da = np.einsum('pi,qi->pq', Ca[:, :nocca], Ca[:, :nocca])
        Db = np.einsum('pi,qi->pq', Ca[:, :noccb], Ca[:, :noccb])

    t = time.time()
    E = 0.0
    Enuc = mol.nuclear_repulsion_energy()
    Eold = 0.0

    Fock_list = []
    DIIS_error = []

    # Build a C matrix and share data with the numpy array npC
    Cocca = psi4.Matrix(nbf, nocca)
    npCa = np.asarray(Cocca)
    npCa[:] = Ca[:, :nocca]

    Coccb = psi4.Matrix(nbf, noccb)
    if noccb > 0:
        npCb = np.asarray(Coccb)
        npCb[:] = Cb[:, :noccb]

    # Initialize the JK object
    psi4.set_global_option('SCF_TYPE', "OUT_OF_CORE")
    jk = psi4.JK.build_JK(wfn.basisset())
    jk.initialize()
    jk.C_left().append(Cocca)
    jk.C_left().append(Coccb)
    jk.print_header()

    # Build a DIIS helper object
    diisa = DIIS_helper()
    diisb = DIIS_helper()

    if do_print:
        print('\nTotal time taken for setup: %.3f seconds' % (time.time() - t))

        print('\nStart SCF iterations:\n')

    t = time.time()

    for SCF_ITER in range(1, maxiter + 1):

        npCa[:] = Ca[:, :nocca]
        if noccb > 0:
            npCb[:] = Cb[:, :noccb]
        #print 'here'
        jk.compute()
        #print 'here'

        # Build fock matrix
        Ja = np.asarray(jk.J()[0])
        Jb = np.asarray(jk.J()[1])
        Ka = np.asarray(jk.K()[0])
        Kb = np.asarray(jk.K()[1])
        Fa = H + (Ja + Jb) - Ka
        Fb = H + (Ja + Jb) - Kb

        # DIIS error build and update
        diisa_e = Fa.dot(Da).dot(S) - S.dot(Da).dot(Fa)
        diisa_e = (A.T).dot(diisa_e).dot(A)

        diisb_e = Fb.dot(Db).dot(S) - S.dot(Db).dot(Fb)
        diisb_e = (A.T).dot(diisb_e).dot(A)

        if SCF_ITER > 1:
            diisa.add(Fa, diisa_e)
            diisb.add(Fb, diisb_e)

        # SCF energy and update
        SCF_E  = np.einsum('pq,pq->', Da + Db, H)
        SCF_E += np.einsum('pq,pq->', Da, Fa)
        SCF_E += np.einsum('pq,pq->', Db, Fb)
        SCF_E *= 0.5
        SCF_E += Enuc

        dRMS = 0.5 * (np.mean(diisa_e**2)**0.5 + np.mean(diisb_e**2)**0.5)

        if do_print:
            print('SCF Iteration %3d: Energy = %20.14f   dE = % 1.5E   dRMS = %1.5E'
                  % (SCF_ITER, SCF_E, (SCF_E - Eold), dRMS))
        if (abs(SCF_E - Eold) < E_conv) and (dRMS < D_conv):
            break

        Eold = SCF_E

        if SCF_ITER > 1:
            Fa = diisa.extrapolate()
            Fb = diisb.extrapolate()

        # Diagonalize Fock matrix
        Ca, Da = diag_H(Fa, nocca, pop_a)
        Cb, Db = diag_H(Fb, noccb, pop_b)

        if SCF_ITER == maxiter:
            clean()
            raise Exception("Maximum number of SCF cycles exceeded.")

    if do_print:
        print('Total time for SCF iterations: %.3f seconds \n' % (time.time() - t))

        spin_mat = (Cb[:, :noccb].T).dot(S).dot(Ca[:, :nocca])
        spin_contam = min(nocca, noccb) - np.vdot(spin_mat, spin_mat)
        print('Spin Contamination Metric: %1.5E\n' % spin_contam)

    ret = {}
    ret["Da"] = Da
    ret["Db"] = Da
    ret["Ca"] = Ca
    ret["Cb"] = Ca
    return ret

def sad(psi4, mol, basis, pop_a=None, pop_b=None, do_print=True, E_conv = 1.e-7, D_conv = 1.e-5):

    # Set defaults
    maxiter = 40
    #E_conv = 1.0E-7
    #D_conv = 1.0E-5

    psi4.set_global_option('BASIS', basis)
    # Integral generation from Psi4's MintsHelper
    t = time.time()
    wfn = psi4.new_wavefunction(mol, psi4.get_global_option('BASIS'))
    mints = psi4.MintsHelper(wfn.basisset())
    S = np.asarray(mints.ao_overlap())

    # Get nbf and ndocc for closed shell molecules
    nbf = S.shape[0]
    if pop_a is not None:
        nocca = pop_a.shape[0]
        if pop_b is None:
            noccb = nocca
            pop_b = noccb
        else:
            noccb = pop_b.shape[0]
    else:
        nocca = wfn.nalpha()
        noccb = wfn.nbeta()

    if mol.label(0) == 'C':
        nocca = 4
        noccb = 2

    if do_print:
        print '\nNumber of alpha orbitals:   %3d' % nocca
        print 'Number of beta orbitals:    %3d' % noccb
        print 'Number of basis functions:  %3d' % nbf

    V = np.asarray(mints.ao_potential())
    T = np.asarray(mints.ao_kinetic())

    if do_print:
        print '\nTotal time taken for integrals: %.3f seconds.' % (time.time()-t)

    t = time.time()

    # Build H_core
    H = T + V

    # Orthogonalizer A = S^(-1/2)
    A = mints.ao_overlap()
    A.power(-0.5, 1.e-16)
    A = np.asarray(A)

    def diag_H(H, nocc, pop=None):
        Hp = A.dot(H).dot(A)
        e, C2 = np.linalg.eigh(Hp)
        C = A.dot(C2)
        Cocc = C[:, :nocc]
        if pop is not None:
            Cocc *= pop
        D = np.einsum('pi,qi->pq', Cocc, Cocc)
        return (C, D)

    Ca, Da = diag_H(H, nocca, pop_a)
    Cb, Db = diag_H(H, noccb, pop_b)

    t = time.time()
    E = 0.0
    Enuc = mol.nuclear_repulsion_energy()
    Eold = 0.0

    Fock_list = []
    DIIS_error = []

    # Initialize the JK object
    I = np.asarray(mints.ao_eri())

    # Build a DIIS helper object
    diisa = DIIS_helper()
    diisb = DIIS_helper()

    if do_print:
        print('\nTotal time taken for setup: %.3f seconds' % (time.time() - t))

        print('\nStart SCF iterations:\n')

    t = time.time()

    for SCF_ITER in range(1, maxiter + 1):


        # Build fock matrix
        Ja = np.einsum('pqrs,rs->pq', I, Da)
        Jb = np.einsum('pqrs,rs->pq', I, Db)
        Ka = np.einsum('prqs,rs->pq', I, Da)
        Kb = np.einsum('prqs,rs->pq', I, Db)
        Fa = H + (Ja + Jb) - Ka
        Fb = H + (Ja + Jb) - Kb

        # DIIS error build and update
        diisa_e = Fa.dot(Da).dot(S) - S.dot(Da).dot(Fa)
        diisa_e = (A.T).dot(diisa_e).dot(A)

        diisb_e = Fb.dot(Db).dot(S) - S.dot(Db).dot(Fb)
        diisb_e = (A.T).dot(diisb_e).dot(A)

        if SCF_ITER > 1:
            diisa.add(Fa, diisa_e)
            diisb.add(Fb, diisb_e)

        # SCF energy and update
        SCF_E  = np.einsum('pq,pq->', Da + Db, H)
        SCF_E += np.einsum('pq,pq->', Da, Fa)
        SCF_E += np.einsum('pq,pq->', Db, Fb)
        SCF_E *= 0.5
        SCF_E += Enuc

        dRMS = 0.5 * (np.mean(diisa_e**2)**0.5 + np.mean(diisb_e**2)**0.5)

        if do_print:
            print('SCF Iteration %3d: Energy = %20.14f   dE = % 1.5E   dRMS = %1.5E'
                  % (SCF_ITER, SCF_E, (SCF_E - Eold), dRMS))
        if (abs(SCF_E - Eold) < E_conv) and (dRMS < D_conv):
            break

        Eold = SCF_E

        if SCF_ITER > 1:
            Fa = diisa.extrapolate()
            Fb = diisb.extrapolate()

        # Diagonalize Fock matrix
        Ca, Da = diag_H(Fa, nocca, pop_a)
        Cb, Db = diag_H(Fb, noccb, pop_b)

        if SCF_ITER == maxiter:
            clean()
            raise Exception("Maximum number of SCF cycles exceeded.")

    if do_print:
        print('Total time for SCF iterations: %.3f seconds \n' % (time.time() - t))

        spin_mat = (Cb[:, :noccb].T).dot(S).dot(Ca[:, :nocca])
        spin_contam = min(nocca, noccb) - np.vdot(spin_mat, spin_mat)
        print('Spin Contamination Metric: %1.5E\n' % spin_contam)

    ret = {}
    ret["Da"] = Da
    ret["Db"] = Da
    ret["Ca"] = Ca
    ret["Cb"] = Ca
    return ret

def basis_projection(C, bas1, bas2):
    mints = MintsHelper(bas1)
    
    SBB = mints.ao_overlap(bas2, bas2).to_array()
    SBA = mints.ao_overlap(bas2, bas1).to_array()
    SAA = mints.ao_overlap(bas1, bas1).to_array()

    CBBinv = np.linalg.inv(SBB)

    T = C.T.dot(SBA.T).dot(CBBinv).dot(SBA).dot(C)
    matT = psi4.Matrix.from_array(T)
    matT.power(-0.5, 1.e-15)
    Cb = CBBinv.dot(SBA).dot(C).dot(matT)
    return Cb
