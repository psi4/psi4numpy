"""
Helper classes and functions for the SCF folder.

References:
- RHF/UHF equations & algorithms from [Szabo:1996]
- DIIS equations & algorithm from [Sherrill:1998], [Pulay:1980:393], & [Pulay:1969:197]
- Orbital rotaion expressions from [Helgaker:2000]
"""

__authors__ = "Daniel G. A. Smith"
__credits__ = ["Daniel G. A. Smith"]

__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__ = "BSD-3-Clause"
__date__ = "2017-9-30"

import time
import numpy as np
import psi4
np.set_printoptions(precision=5, linewidth=200, suppress=True)


class helper_HF(object):
    """
    A generalized Hartree-Fock helper script.

    Notes
    -----
    Equations and algorithms from [Szabo:1996]
    """

    def __init__(self, mol, basis=None, memory=2, ndocc=None, scf_type='DF', guess='CORE'):
        """
        Initializes the helper_HF object.

        Parameters
        ----------
        mol : psi4.core.Molecule or str
            The molecule to be used for the given helper object.
        basis : {str, None}, optional
            The basis string to be used
        memory : {int, 2}, optional
            The amount of memory (in GB) to use.
        ndocc : {int, None}, optional
            The number of occupied orbitals for the HF helper. Defaults to the number of electrons divided by 2.
        scf_type : {"DF", "PK", "OUT_OF_CORE"}, optional
            The type of JK object to use.
        guess : {"CORE", "SAD"}, optional
            The initial guess type to attempt.

        Returns
        ------
        ret : helper_HF
            A initialized helper_HF object

        Examples
        --------

        # Construct the helper object
        >>> helper = helper_HF("He\nHe 1 2.0", "cc-pVDZ")

        # Take a Roothan-Hall step
        >>> F = helper.build_fock()
        >>> helper.compute_hf_energy()
        -5.4764571474633197


        # Take a Roothan-Hall step
        >>> e, C = helper.diag(F)
        >>> helper.set_Cleft(C)
        >>> F = helper.build_fock()
        >>> helper.compute_hf_energy()
        -5.706674424039214

        """

        # Build and store all 2D values
        print('Building rank 2 integrals...')
        t = time.time()

        if not isinstance(mol, psi4.core.Molecule):
            mol = psi4.geometry(mol)

        if basis is None:
            basis = psi4.core.get_global_option('BASIS')
        else:
            psi4.core.set_global_option("BASIS", basis)

        wfn = psi4.core.Wavefunction.build(mol, basis)
        self.wfn = wfn
        self.mints = psi4.core.MintsHelper(wfn.basisset())
        self.enuc = mol.nuclear_repulsion_energy()

        # Build out necessary 2D matrices
        self.S = np.asarray(self.mints.ao_overlap())
        self.V = np.asarray(self.mints.ao_potential())
        self.T = np.asarray(self.mints.ao_kinetic())
        self.H = self.T + self.V

        # Holder objects
        self.Da = None
        self.Db = None
        self.Ca = None
        self.Cb = None

        self.J = None
        self.K = None

        # Build symmetric orthoganlizer
        A = self.mints.ao_overlap()
        A.power(-0.5, 1.e-14)
        self.A = np.asarray(A)

        # Get nbf and ndocc for closed shell molecules
        self.epsilon = None
        self.nbf = self.S.shape[0]
        if ndocc:
            self.ndocc = ndocc
        else:
            self.ndocc = int(sum(mol.Z(A) for A in range(mol.natom())) / 2)

        # Only rhf for now
        self.nvirt = self.nbf - self.ndocc

        print('\nNumber of occupied orbitals: %d' % self.ndocc)
        print('Number of basis functions: %d' % self.nbf)

        self.C_left = psi4.core.Matrix(self.nbf, self.ndocc)
        self.npC_left = np.asarray(self.C_left)

        if guess.upper() == 'CORE':
            Xp = self.A.dot(self.H).dot(self.A)
            e, C2 = np.linalg.eigh(Xp)

            self.Ca = self.A.dot(C2)
            self.npC_left[:] = self.Ca[:, :self.ndocc]
            self.epsilon = e
            self.Da = np.dot(self.npC_left, self.npC_left.T)
            self.F = self.H

        elif guess.upper() == 'SAD':

            # Cheat and run a quick SCF calculation
            psi4.set_options({'E_CONVERGENCE': 1, 'D_CONVERGENCE': 1})

            e, wfn = psi4.energy('SCF', return_wfn=True)
            self.Ca = np.array(wfn.Ca())
            self.npC_left[:] = self.Ca[:, :self.ndocc]
            self.epsilon = np.array(wfn.epsilon_a())
            self.Da = np.dot(self.npC_left, self.npC_left.T)

            psi4.set_options({'E_CONVERGENCE': 6, 'D_CONVERGENCE': 6})

        else:
            raise Exception("Guess %s not yet supported" % (guess))

        self.DIIS_error = []
        self.DIIS_F = []

        scf_type = scf_type.upper()
        if scf_type not in ['DF', 'PK', 'DIRECT', 'OUT_OF_CORE']:
            raise Exception('SCF_TYPE %s not supported' % scf_type)
        psi4.set_options({'SCF_TYPE': scf_type})
        self.jk = psi4.core.JK.build(wfn.basisset())
        self.jk.initialize()
        #        self.jk.C_left().append(self.C_left)

        print('...rank 2 integrals built in %.3f seconds.' % (time.time() - t))

    def set_Cleft(self, C):
        """
        Sets the current orbital matrix and builds the density from it.
        """

        if (C.shape[1] == self.ndocc):
            Cocc = C
        elif (C.shape[1] == self.nbf):
            self.Ca = C
            Cocc = C[:, :self.ndocc]
        else:
            raise Exception("Cocc shape is %s, need %s." % (str(self.npC_left.shape), str(Cocc.shape)))
        self.npC_left[:] = Cocc
        self.Da = np.dot(Cocc, Cocc.T)

    def diag(self, F, set_C=False):
        """
        Diagonalizes the matrix F using the symmetric orthogonalization matrix S^{-1/2}.

        Parameters
        ----------
        F : numpy.array
            Fock matrix to diagonalize according to [Szabo:1996] pp. 145
        set_C : {True, False}, optional
            Set the computed C matrix as the Cleft attribute?

        Returns
        -------
        e : numpy.array
            Array of orbital energies (eigenvalues of Fock matrix)
        C : numpy.array
            Orbital coefficient matrix
        """
        Xp = self.A.dot(F).dot(self.A)
        e, C2 = np.linalg.eigh(Xp)
        C = self.A.dot(C2)
        if set_C:
            self.set_Cleft(C)
        return e, C

    def build_fock(self):
        """
        Builds the Fock matrix from the current orbitals

        D = Cocc Cocc.T
        F = H + 2 * J[D] - K[D]
        """

        self.jk.C_left_add(self.C_left)
        self.jk.compute()
        self.jk.C_clear()
        self.J = np.asarray(self.jk.J()[0])
        self.K = np.asarray(self.jk.K()[0])
        self.F = self.H + self.J * 2 - self.K
        return self.F

    def build_jk(self, C_left, C_right=None):
        """
        A wrapper to compute the J and K objects.
        """
        return compute_jk(self.jk, C_left, C_right)

    def compute_hf_energy(self):
        """
        Computes the current SCF energy (F + H)_pq D_pq + E_nuclear
        """
        self.scf_e = np.einsum('ij,ij->', self.F + self.H, self.Da) + self.enuc
        return self.scf_e


class DIIS_helper(object):
    """
    A helper class to compute DIIS extrapolations.

    Notes
    -----
    Equations taken from [Sherrill:1998], [Pulay:1980:393], & [Pulay:1969:197]
    Algorithms adapted from [Sherrill:1998] & [Pulay:1980:393]
    """

    def __init__(self, max_vec=6):
        """
        Intializes the DIIS class.

        Parameters
        ----------
        max_vec : int (default, 6)
            The maximum number of vectors to use. The oldest vector will be deleted.
        """
        self.error = []
        self.vector = []
        self.max_vec = max_vec

    def add(self, state, error):
        """
        Adds a set of error and state vectors to the DIIS object.

        Parameters
        ----------
        state : array_like
            The state vector to add to the DIIS object.
        error : array_like
            The error vector to add to the DIIS object.

        Returns
        ------
        None
        """

        error = np.array(error)
        state = np.array(state)
        if len(self.error) > 1:
            if self.error[-1].shape[0] != error.size:
                raise Exception("Error vector size does not match previous vector.")
            if self.vector[-1].shape != state.shape:
                raise Exception("Vector shape does not match previous vector.")

        self.error.append(error.ravel().copy())
        self.vector.append(state.copy())

    def extrapolate(self):
        """
        Performs the DIIS extrapolation for the objects state and error vectors.

        Parameters
        ----------
        None

        Returns
        ------
        ret : ndarray
            The extrapolated next state vector

        """

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


def compute_jk(jk, C_left, C_right=None):
    """
    A python wrapper for a Psi4 JK object to consume and produce NumPy arrays.

    Computes the following matrices:
    D = C_left C_right.T
    J_pq = (pq|rs) D_rs
    K_pq = (pr|qs) D_rs

    Parameters
    ----------
    jk : psi4.core.JK
        A initialized Psi4 JK object
    C_left : list of array_like or a array_like object
        Orbitals used to compute the JK object with
    C_right : list of array_like (optional, None)
        Optional C_right orbitals, otherwise it is assumed C_right == C_left

    Returns
    -------
    JK : tuple of ndarray
        Returns the J and K objects

    Notes
    -----
    This function uses the Psi4 JK object and will compute the initialized JK type (DF, PK, OUT_OF_CORE, etc)


    Examples
    --------

    ndocc = 5
    nbf = 15

    Cocc = np.random.rand(nbf, ndocc)

    jk = psi4.core.JK.build(wfn.basisset())
    jk.set_memory(int(1.25e8))  # 1GB
    jk.initialize()
    jk.print_header()


    J, K = compute_jk(jk, Cocc)

    J_list, K_list = compute_jk(jk, [Cocc, Cocc])
    """

    # Clear out the matrices
    jk.C_clear()

    list_input = True
    if not isinstance(C_left, (list, tuple)):
        C_left = [C_left]
        list_input = False

    for c in C_left:
        mat = psi4.core.Matrix.from_array(c)
        jk.C_left_add(mat)

    # Do we have C_right?
    if C_right is not None:
        if not isinstance(C_right, (list, tuple)):
            C_right = [C_right]

        if len(C_left) != len(C_right):
            raise ValueError("JK: length of left and right matrices is not equal")

        if not isinstance(C_right, (list, tuple)):
            C_right = [C_right]

        for c in C_right:
            mat = psi4.core.Matrix.from_array(c)
            jk.C_right_add(mat)

    # Compute the JK
    jk.compute()

    # Unpack
    J = []
    K = []
    for n in range(len(C_left)):
        J.append(np.array(jk.J()[n]))
        K.append(np.array(jk.K()[n]))

    jk.C_clear()

    # Duck type the return
    if list_input:
        return (J, K)
    else:
        return (J[0], K[0])


def rotate_orbitals(C, x, return_d=False):
    """
    Rotates the orbitals C using the rotation matrix x.

    Using the antisymmetric skew Matrix: U = [x,  0]
                                             [0, -x]

    C' = C e^{U}

    Parameters
    ----------
    C : array_like
        The orbital matrix to rotate
    x : array_like
        The occupied by virtual orbital rotation matrix
    return_d : bool (optional, False)
        Returns the occupied Density matrix of the rotated orbitals if requested

    Returns
    -------
    C' : ndarray
        The rotated orbital matrix

    Notes
    -----
    This function uses a truncated Taylor expansion to approximate the exponential:
        e^{U} \approx 1 + U + 0.5 * U U

    Equations from [Helgaker:2000]

    Examples
    --------

    ndocc = 5
    nvir = 10
    nbf = ndocc + nvir

    C = np.random.rand(nbf, nbf)
    x = np.random.rand(ndocc, nvir)

    Cp = rotate_orbitals(C, x)
    """

    # NumPy Array conversion
    C = np.asarray(C)
    x = np.asarray(x)

    rsize = x.shape[0] + x.shape[1]
    if (rsize) != C.shape[1]:
        raise ValueError("rotate_orbitals: shape mismatch")

    # Build U
    U = np.zeros((rsize, rsize))
    ndocc = x.shape[0]
    U[:ndocc, ndocc:] = x
    U[ndocc:, :ndocc] = -x.T

    U += 0.5 * np.dot(U, U)
    U[np.diag_indices_from(U)] += 1

    # Easy access to Schmidt orthogonalization
    U, r = np.linalg.qr(U.T)

    # Rotate and set orbitals
    C = C.dot(U)
    if return_d:
        Cocc = C[:, :ndocc]
        return C, np.dot(Cocc, Cocc.T)
    else:
        return C


def transform_aotoso(m_ao, transformers):
    """
    Transform an operator from the atomic orbital to spin orbital basis.

    Parameters
    ----------
    m_ao : numpy.ndarray
        A [nao, nao] matrix
    transformers : list or tuple of numpy.ndarray
        Transformation matrices, one for each irrep, with shape [nao, nso in irrep]

    Returns
    -------
    tuple of numpy.ndarray
        One matrix for each irrep with shape [nso in irrep, nso in irrep]
    """
    return tuple(transformer.T.dot(m_ao).dot(transformer)
                 for transformer in transformers)


def transform_sotoao(m_so_, transformers):
    """
    Transform an operator from the spin orbital to the atomic orbital basis.

    Parameters
    ----------
    m_so_ : list or tuple of numpy.ndarray
        Matrices, one for each irrep, with shape [nso in irrep, nso in irrep]
    transformers : list or tuple of numpy.ndarray
        Transformation matrices, one for each irrep, with shape [nao, nso in irrep]

    Returns
    -------
    numpy.ndarray
        A [nao, nao] matrix
    """
    assert len(m_so_) == len(transformers)
    return sum(transformer.dot(m_so).dot(transformer.T)
               for transformer, m_so in zip(transformers, m_so_))
