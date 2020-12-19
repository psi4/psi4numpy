"""
Returns the Fock matrix and needed two-electron integral blocks.

__authors__   =  "Jonathon P. Misiewicz"
__credits__   =  ["Jonathon P. Misiewicz"]

__copyright__ = "(c) 2014-2020, The Psi4NumPy Developers"
__license__   = "BSD-3-Clause"
"""

import numpy as np
import psi4


def orbitals_fock(mol, singles):
    wfn = psi4.energy('scf', molecule=mol, return_wfn=True)[1]

    ### Orbitals
    CA_O = wfn.Ca_subset("AO", "ACTIVE_OCC")
    CB_O = wfn.Cb_subset("AO", "ACTIVE_OCC")
    C_O = np.block([[CA_O, np.zeros(CB_O.shape)], [np.zeros(CA_O.shape), CB_O]])
    CA_V = wfn.Ca_subset("AO", "ACTIVE_VIR")
    CB_V = wfn.Cb_subset("AO", "ACTIVE_VIR")
    C_V = np.block([[CA_V, np.zeros(CB_V.shape)], [np.zeros(CA_V.shape), CB_V]])
    mints = psi4.core.MintsHelper(wfn.basisset())

    ### One-Electron Integrals
    Fa = wfn.Fa()
    Fb = wfn.Fb()
    FI = np.block([[Fa, np.zeros(Fb.shape)], [np.zeros(Fa.shape), Fb]])
    F = {
        "oo": np.einsum('pP,qQ,pq->PQ', C_O, C_O, FI, optimize=True),
        "vv": np.einsum('pP,qQ,pq->PQ', C_V, C_V, FI, optimize=True)
    }
    if singles:
        F["ov"] = np.einsum('pP, qQ, pq -> PQ', C_O, C_V, FI, optimize=True)

    return C_O, C_V, mints, F


def integrals(mol, singles=False, return_intermediates=False):

    C_O, C_V, mints, F = orbitals_fock(mol, singles)

    ### Two-Electron Integrals
    TEI = mints.ao_eri().np
    # Construct electron-repulsion integrals in spinorbital basis from spatial orbital basis.
    TEI = np.kron(np.eye(2), np.kron(np.eye(2), TEI).T)
    # Transform integrals to physicist notation
    TEI = TEI.swapaxes(1, 2)
    # Antisymmetrize...
    TEI -= TEI.swapaxes(2, 3)
    I = {
        "oovv": np.einsum('pP,qQ,rR,sS,pqrs->PQRS', C_O, C_O, C_V, C_V, TEI, optimize=True),
        "oooo": np.einsum('pP,qQ,rR,sS,pqrs->PQRS', C_O, C_O, C_O, C_O, TEI, optimize=True),
        "voov": np.einsum('pP,qQ,rR,sS,pqrs->PQRS', C_V, C_O, C_O, C_V, TEI, optimize=True),
        "vvvv": np.einsum('pP,qQ,rR,sS,pqrs->PQRS', C_V, C_V, C_V, C_V, TEI, optimize=True)
    }

    if singles:
        I["ovvv"] = np.einsum('pP,qQ,rR,sS,pqrs->PQRS', C_O, C_V, C_V, C_V, TEI, optimize=True)
        I["ooov"] = np.einsum('pP,qQ,rR,sS,pqrs->PQRS', C_O, C_O, C_O, C_V, TEI, optimize=True)

    if not return_intermediates:
        return I, F
    else:
        OEI = np.kron(np.eye(2), mints.ao_kinetic().np + mints.ao_potential().np)
        intermed = {"TEI": TEI, "O": C_O, "V": C_V, "OEI": OEI}
        return I, F, intermed


def integrals_DF(mol, singles=False):

    C_O, C_V, mints, F = orbitals_fock(mol, singles)

    ### Two-Electron Integrals
    basis = mints.basisset()
    zero_bas = psi4.core.BasisSet.zero_ao_basis_set()
    aux = psi4.core.BasisSet.build(mol, "DF_BASIS_CC", "", "RIFIT", basis.name())
    Ppq = np.squeeze(mints.ao_eri(aux, zero_bas, basis, basis))
    Ppq = np.kron(np.eye(2), Ppq)  # Construct spinorbital quantities
    metric = mints.ao_eri(aux, zero_bas, aux, zero_bas)
    metric.power(-0.5, 1.e-10)
    metric = np.squeeze(metric)
    Qpq = np.einsum("QP,Ppq->pqQ", metric, Ppq, optimize=True)
    R = {
        "oo": np.einsum('pP,qQ,pqX->PQX', C_O, C_O, Qpq, optimize=True),
        "ov": np.einsum('pP,qQ,pqX->PQX', C_O, C_V, Qpq, optimize=True),
        "vv": np.einsum('pP,qQ,pqX->PQX', C_V, C_V, Qpq, optimize=True)
    }

    return R, F
