"""
Helper function for CQED_RHF

References:
    Equations and algorithms from 
    [Haugland:2020:041043], [DePrince:2021:094112], and [McTague:2021:ChemRxiv] 

"""

__authors__ = ["Jon McTague", "Jonathan Foley"]
__credits__ = ["Jon McTague", "Jonathan Foley"]

__copyright_amp__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__ = "BSD-3-Clause"
__date__ = "2021-08-19"

# ==> Import Psi4, NumPy, & SciPy <==
import psi4
import numpy as np
import time


def cqed_rhf(lambda_vector, molecule_string, psi4_options_dict):
    """Computes the QED-RHF energy and density

    Arguments
    ---------
    lambda_vector : 1 x 3 array of floats
        the electric field vector, see e.g. Eq. (1) in [DePrince:2021:094112]
        and (15) in [Haugland:2020:041043]

    molecule_string : string
        specifies the molecular geometry

    options_dict : dictionary
        specifies the psi4 options to be used in running the canonical RHF

    Returns
    -------
    cqed_rhf_dictionary : dictionary
        Contains important quantities from the cqed_rhf calculation, with keys including:
            'RHF ENERGY' -> result of canonical RHF calculation using psi4 defined by molecule_string and psi4_options_dict
            'CQED-RHF ENERGY' -> result of CQED-RHF calculation, see Eq. (13) of [McTague:2021:ChemRxiv]
            'CQED-RHF C' -> orbitals resulting from CQED-RHF calculation
            'CQED-RHF DENSITY MATRIX' -> density matrix resulting from CQED-RHF calculation
            'CQED-RHF EPS'  -> orbital energies from CQED-RHF calculation
            'PSI4 WFN' -> wavefunction object from psi4 canonical RHF calcluation
            'CQED-RHF DIPOLE MOMENT' -> total dipole moment from CQED-RHF calculation (1x3 numpy array)
            'NUCLEAR DIPOLE MOMENT' -> nuclear dipole moment (1x3 numpy array)
            'DIPOLE ENERGY' -> See Eq. (14) of [McTague:2021:ChemRxiv]
            'NUCLEAR REPULSION ENERGY' -> Total nuclear repulsion energy

    Example
    -------
    >>> cqed_rhf_dictionary = cqed_rhf([0., 0., 1e-2], '''\nMg\nH 1 1.7\nsymmetry c1\n1 1\n''', psi4_options_dictionary)

    """
    # define geometry using the molecule_string
    mol = psi4.geometry(molecule_string)
    # define options for the calculation
    psi4.set_options(psi4_options_dict)
    # run psi4 to get ordinary scf energy and wavefunction object
    psi4_rhf_energy, wfn = psi4.energy("scf", return_wfn=True)

    # Create instance of MintsHelper class
    mints = psi4.core.MintsHelper(wfn.basisset())

    # Grab data from wavfunction
    # number of doubly occupied orbitals
    ndocc = wfn.nalpha()

    # grab all transformation vectors and store to a numpy array
    C = np.asarray(wfn.Ca())

    # use canonical RHF orbitals for guess CQED-RHF orbitals
    Cocc = C[:, :ndocc]

    # form guess density
    D = np.einsum("pi,qi->pq", Cocc, Cocc)  # [Szabo:1996] Eqn. 3.145, pp. 139

    # Integrals required for CQED-RHF
    # Ordinary integrals first
    V = np.asarray(mints.ao_potential())
    T = np.asarray(mints.ao_kinetic())
    I = np.asarray(mints.ao_eri())

    # Extra terms for Pauli-Fierz Hamiltonian
    # nuclear dipole
    mu_nuc_x = mol.nuclear_dipole()[0]
    mu_nuc_y = mol.nuclear_dipole()[1]
    mu_nuc_z = mol.nuclear_dipole()[2]

    # electronic dipole integrals in AO basis
    mu_ao_x = np.asarray(mints.ao_dipole()[0])
    mu_ao_y = np.asarray(mints.ao_dipole()[1])
    mu_ao_z = np.asarray(mints.ao_dipole()[2])

    # \lambda \cdot \mu_el (see within the sum of line 3 of Eq. (9) in [McTague:2021:ChemRxiv])
    l_dot_mu_el = lambda_vector[0] * mu_ao_x
    l_dot_mu_el += lambda_vector[1] * mu_ao_y
    l_dot_mu_el += lambda_vector[2] * mu_ao_z

    # compute electronic dipole expectation value with
    # canonincal RHF density
    mu_exp_x = np.einsum("pq,pq->", 2 * mu_ao_x, D)
    mu_exp_y = np.einsum("pq,pq->", 2 * mu_ao_y, D)
    mu_exp_z = np.einsum("pq,pq->", 2 * mu_ao_z, D)

    # need to add the nuclear term to the sum over the electronic dipole integrals
    mu_exp_x += mu_nuc_x
    mu_exp_y += mu_nuc_y
    mu_exp_z += mu_nuc_z

    rhf_dipole_moment = np.array([mu_exp_x, mu_exp_y, mu_exp_z])

    # We need to carry around the electric field dotted into the nuclear dipole moment
    # and the electric field dotted into the RHF electronic dipole expectation value
    # see prefactor to sum of Line 3 of Eq. (9) in [McTague:2021:ChemRxiv]

    # \lambda_vector \cdot \mu_{nuc}
    l_dot_mu_nuc = (
        lambda_vector[0] * mu_nuc_x
        + lambda_vector[1] * mu_nuc_y
        + lambda_vector[2] * mu_nuc_z
    )
    # \lambda_vecto \cdot < \mu > where <\mu> contains electronic and nuclear contributions
    l_dot_mu_exp = (
        lambda_vector[0] * mu_exp_x
        + lambda_vector[1] * mu_exp_y
        + lambda_vector[2] * mu_exp_z
    )

    # dipole energy, Eq. (14) in [McTague:2021:ChemRxiv]
    #  0.5 * (\lambda_vector \cdot \mu_{nuc})** 2
    #      - (\lambda_vector \cdot <\mu> ) ( \lambda_vector\cdot \mu_{nuc})
    # +0.5 * (\lambda_vector \cdot <\mu>) ** 2
    d_c = (
        0.5 * l_dot_mu_nuc ** 2 - l_dot_mu_nuc * l_dot_mu_exp + 0.5 * l_dot_mu_exp ** 2
    )

    # quadrupole arrays
    Q_ao_xx = np.asarray(mints.ao_quadrupole()[0])
    Q_ao_xy = np.asarray(mints.ao_quadrupole()[1])
    Q_ao_xz = np.asarray(mints.ao_quadrupole()[2])
    Q_ao_yy = np.asarray(mints.ao_quadrupole()[3])
    Q_ao_yz = np.asarray(mints.ao_quadrupole()[4])
    Q_ao_zz = np.asarray(mints.ao_quadrupole()[5])

    # Pauli-Fierz 1-e quadrupole terms, Line 2 of Eq. (9) in [McTague:2021:ChemRxiv]
    Q_PF = -0.5 * lambda_vector[0] * lambda_vector[0] * Q_ao_xx
    Q_PF -= 0.5 * lambda_vector[1] * lambda_vector[1] * Q_ao_yy
    Q_PF -= 0.5 * lambda_vector[2] * lambda_vector[2] * Q_ao_zz

    # accounting for the fact that Q_ij = Q_ji
    # by weighting Q_ij x 2 which cancels factor of 1/2
    Q_PF -= lambda_vector[0] * lambda_vector[1] * Q_ao_xy
    Q_PF -= lambda_vector[0] * lambda_vector[2] * Q_ao_xz
    Q_PF -= lambda_vector[1] * lambda_vector[2] * Q_ao_yz

    # Pauli-Fierz 1-e dipole terms scaled by
    # (\lambda_vector \cdot \mu_{nuc} - \lambda_vector \cdot <\mu>)
    # Line 3 in full of Eq. (9) in [McTague:2021:ChemRxiv]
    d_PF = (l_dot_mu_nuc - l_dot_mu_exp) * l_dot_mu_el

    # ordinary H_core
    H_0 = T + V

    # Add Pauli-Fierz terms to H_core
    # Eq. (11) in [McTague:2021:ChemRxiv]
    H = H_0 + Q_PF + d_PF

    # Overlap for DIIS
    S = mints.ao_overlap()
    # Orthogonalizer A = S^(-1/2) using Psi4's matrix power.
    A = mints.ao_overlap()
    A.power(-0.5, 1.0e-16)
    A = np.asarray(A)

    print("\nStart SCF iterations:\n")
    t = time.time()
    E = 0.0
    Enuc = mol.nuclear_repulsion_energy()
    Eold = 0.0
    E_1el_crhf = np.einsum("pq,pq->", H_0 + H_0, D)
    E_1el = np.einsum("pq,pq->", H + H, D)
    print("Canonical RHF One-electron energy = %4.16f" % E_1el_crhf)
    print("CQED-RHF One-electron energy      = %4.16f" % E_1el)
    print("Nuclear repulsion energy          = %4.16f" % Enuc)
    print("Dipole energy                     = %4.16f" % d_c)

    # Set convergence criteria from psi4_options_dict
    if "e_convergence" in psi4_options_dict:
        E_conv = psi4_options_dict["e_convergence"]
    else:
        E_conv = 1.0e-7
    if "d_convergence" in psi4_options_dict:
        D_conv = psi4_options_dict["d_convergence"]
    else:
        D_conv = 1.0e-5

    t = time.time()

    # maxiter
    maxiter = 500
    for SCF_ITER in range(1, maxiter + 1):

        # Build fock matrix: [Szabo:1996] Eqn. 3.154, pp. 141
        J = np.einsum("pqrs,rs->pq", I, D)
        K = np.einsum("prqs,rs->pq", I, D)

        # Pauli-Fierz 2-e dipole-dipole terms, line 2 of Eq. (12) in [McTague:2021:ChemRxiv]
        M = np.einsum("pq,rs,rs->pq", l_dot_mu_el, l_dot_mu_el, D)
        N = np.einsum("pr,qs,rs->pq", l_dot_mu_el, l_dot_mu_el, D)

        # Build fock matrix: [Szabo:1996] Eqn. 3.154, pp. 141
        # plus Pauli-Fierz terms Eq. (12) in [McTague:2021:ChemRxiv]
        F = H + J * 2 - K + 2 * M - N

        diis_e = np.einsum("ij,jk,kl->il", F, D, S) - np.einsum("ij,jk,kl->il", S, D, F)
        diis_e = A.dot(diis_e).dot(A)
        dRMS = np.mean(diis_e ** 2) ** 0.5

        # SCF energy and update: [Szabo:1996], Eqn. 3.184, pp. 150
        # Pauli-Fierz terms Eq. 13 of [McTague:2021:ChemRxiv]
        SCF_E = np.einsum("pq,pq->", F + H, D) + Enuc + d_c

        print(
            "SCF Iteration %3d: Energy = %4.16f   dE = % 1.5E   dRMS = %1.5E"
            % (SCF_ITER, SCF_E, (SCF_E - Eold), dRMS)
        )
        if (abs(SCF_E - Eold) < E_conv) and (dRMS < D_conv):
            break

        Eold = SCF_E

        # Diagonalize Fock matrix: [Szabo:1996] pp. 145
        Fp = A.dot(F).dot(A)  # Eqn. 3.177
        e, C2 = np.linalg.eigh(Fp)  # Solving Eqn. 1.178
        C = A.dot(C2)  # Back transform, Eqn. 3.174
        Cocc = C[:, :ndocc]
        D = np.einsum("pi,qi->pq", Cocc, Cocc)  # [Szabo:1996] Eqn. 3.145, pp. 139

        # update electronic dipole expectation value
        mu_exp_x = np.einsum("pq,pq->", 2 * mu_ao_x, D)
        mu_exp_y = np.einsum("pq,pq->", 2 * mu_ao_y, D)
        mu_exp_z = np.einsum("pq,pq->", 2 * mu_ao_z, D)

        mu_exp_x += mu_nuc_x
        mu_exp_y += mu_nuc_y
        mu_exp_z += mu_nuc_z

        # update \lambda \cdot <\mu>
        l_dot_mu_exp = (
            lambda_vector[0] * mu_exp_x
            + lambda_vector[1] * mu_exp_y
            + lambda_vector[2] * mu_exp_z
        )
        # Line 3 in full of Eq. (9) in [McTague:2021:ChemRxiv]
        d_PF = (l_dot_mu_nuc - l_dot_mu_exp) * l_dot_mu_el

        # update Core Hamiltonian
        H = H_0 + Q_PF + d_PF

        # update dipole energetic contribution, Eq. (14) in [McTague:2021:ChemRxiv]
        d_c = (
            0.5 * l_dot_mu_nuc ** 2
            - l_dot_mu_nuc * l_dot_mu_exp
            + 0.5 * l_dot_mu_exp ** 2
        )

        if SCF_ITER == maxiter:
            psi4.core.clean()
            raise Exception("Maximum number of SCF cycles exceeded.")

    print("Total time for SCF iterations: %.3f seconds \n" % (time.time() - t))
    print("QED-RHF   energy: %.8f hartree" % SCF_E)
    print("Psi4  SCF energy: %.8f hartree" % psi4_rhf_energy)

    rhf_one_e_cont = (
        2 * H_0
    )  # note using H_0 which is just T + V, and does not include Q_PF and d_PF
    rhf_two_e_cont = (
        J * 2 - K
    )  # note using just J and K that would contribute to ordinary RHF 2-electron energy
    pf_two_e_cont = 2 * M - N

    SCF_E_One = np.einsum("pq,pq->", rhf_one_e_cont, D)
    SCF_E_Two = np.einsum("pq,pq->", rhf_two_e_cont, D)
    CQED_SCF_E_Two = np.einsum("pq,pq->", pf_two_e_cont, D)

    CQED_SCF_E_D_PF = np.einsum("pq,pq->", 2 * d_PF, D)
    CQED_SCF_E_Q_PF = np.einsum("pq,pq->", 2 * Q_PF, D)

    assert np.isclose(
        SCF_E_One + SCF_E_Two + CQED_SCF_E_D_PF + CQED_SCF_E_Q_PF + CQED_SCF_E_Two,
        SCF_E - d_c - Enuc,
    )

    cqed_rhf_dict = {
        "RHF ENERGY": psi4_rhf_energy,
        "CQED-RHF ENERGY": SCF_E,
        "1E ENERGY": SCF_E_One,
        "2E ENERGY": SCF_E_Two,
        "1E DIPOLE ENERGY": CQED_SCF_E_D_PF,
        "1E QUADRUPOLE ENERGY": CQED_SCF_E_Q_PF,
        "2E DIPOLE ENERGY": CQED_SCF_E_Two,
        "CQED-RHF C": C,
        "CQED-RHF DENSITY MATRIX": D,
        "CQED-RHF EPS": e,
        "PSI4 WFN": wfn,
        "RHF DIPOLE MOMENT": rhf_dipole_moment,
        "CQED-RHF DIPOLE MOMENT": np.array([mu_exp_x, mu_exp_y, mu_exp_z]),
        "NUCLEAR DIPOLE MOMENT": np.array([mu_nuc_x, mu_nuc_y, mu_nuc_z]),
        "DIPOLE ENERGY": d_c,
        "NUCLEAR REPULSION ENERGY": Enuc,
    }

    return cqed_rhf_dict
