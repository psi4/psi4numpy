"""
Computes the Orbital-Optimized LCCD correlation energy.
Equations taken from [Bozkaya:2013:054104].

__authors__   =  "Jonathon P. Misiewicz"
__credits__   =  ["Jonathon P. Misiewicz"]

__copyright__ = "(c) 2014-2020, The Psi4NumPy Developers"
__license__   = "BSD-3-Clause"
"""

import numpy as np
import psi4
from scipy import linalg as spla
from DSD import DirectSumDiis
from integrals import integrals

### Settings

mol = psi4.geometry("""
O
H 1 1.0
H 1 1.0 2 104.5
symmetry c1""")
scf_type = "pk"
target_convergence = 7
maxiter = 50
compare_psi4 = True
basis = "sto-3g"
freeze_core = True

### Setup

if freeze_core and compare_psi4:
    raise Exception("fc-OLCCD not yet implemented. :(")

psi4.set_options({"scf_type": scf_type, "e_convergence": target_convergence + 1, "basis": basis, "freeze_core": freeze_core})
I, F, intermed = integrals(mol, singles=True, return_intermediates=True)
t_ov = np.zeros(F["ov"].shape)
t_cv = np.zeros(F["cv"].shape)
t_co = np.zeros(F["co"].shape)
t2 = np.zeros(I["oovv"].shape)
dsd = DirectSumDiis(3, 8)
num_occ, num_vir = t_ov.shape
num_cor, _ = t_cv.shape

def SD_E(mol, H, I):
    return mol.nuclear_repulsion_energy() + np.trace(H["oo"]) + np.trace(H["cc"]) + 0.5 * np.einsum("ijij ->", I["oooo"], optimize = True) + 0.5 * np.einsum("cdcd ->", I["cccc"], optimize = True) + np.einsum("cici ->", I["coco"], optimize = True)

# We need the initial one-electron integrals as well for orbital-optimized methods.
# The orbital gradient expression requires these instead of the Fock integrals that our amplitude expressions need.
H = {(x + y).lower(): np.einsum('pP, qQ, pq -> PQ', intermed[x], intermed[y], intermed["OEI"], optimize = True) for x, y in ["CO", "CV", "CC", "OO", "OV", "VV"]}
Escf = SD_E(mol, H, I)

### Main Loop
for i in range(1, maxiter + 1):

    if i != 1:
        # Compute new orbitals and transform into them. See Section IIA5 of Bozkaya.
        Zcc = np.zeros((num_cor, num_cor))
        Zoo = np.zeros((num_occ, num_occ))
        Zvv = np.zeros((num_vir, num_vir))
        X = np.block([[Zcc,  -t_co, -t_cv],
                      [t_co.T,  Zoo, -t_ov],
                     [t_cv.T, t_ov.T, Zvv]])
        U = spla.expm(X)
        C = np.hstack((intermed["C"], intermed["O"], intermed["V"]))
        C = C @ U
        C_C, C_O, C_V = np.hsplit(C, [num_cor, num_cor + num_occ])
        I = {
            "oovv": np.einsum('pP,qQ,rR,sS,pqrs->PQRS', C_O, C_O, C_V, C_V, intermed["TEI"], optimize = True),
            "oooo": np.einsum('pP,qQ,rR,sS,pqrs->PQRS', C_O, C_O, C_O, C_O, intermed["TEI"], optimize = True),
            "voov": np.einsum('pP,qQ,rR,sS,pqrs->PQRS', C_V, C_O, C_O, C_V, intermed["TEI"], optimize = True),
            "vvvv": np.einsum('pP,qQ,rR,sS,pqrs->PQRS', C_V, C_V, C_V, C_V, intermed["TEI"], optimize = True),
            "ovvv": np.einsum('pP,qQ,rR,sS,pqrs->PQRS', C_O, C_V, C_V, C_V, intermed["TEI"], optimize = True),
            "ooov": np.einsum('pP,qQ,rR,sS,pqrs->PQRS', C_O, C_O, C_O, C_V, intermed["TEI"], optimize = True),
            "coco": np.einsum('pP,qQ,rR,sS,pqrs->PQRS', C_C, C_O, C_C, C_O, intermed["TEI"], optimize = True),
            "cocv": np.einsum('pP,qQ,rR,sS,pqrs->PQRS', C_C, C_O, C_C, C_V, intermed["TEI"], optimize = True),
            "cvcv": np.einsum('pP,qQ,rR,sS,pqrs->PQRS', C_C, C_V, C_C, C_V, intermed["TEI"], optimize = True),
            "ccco": np.einsum('pP,qQ,rR,sS,pqrs->PQRS', C_C, C_C, C_C, C_O, intermed["TEI"], optimize = True),
            "coov": np.einsum('pP,qQ,rR,sS,pqrs->PQRS', C_C, C_O, C_O, C_V, intermed["TEI"], optimize = True),
            "cooo": np.einsum('pP,qQ,rR,sS,pqrs->PQRS', C_C, C_O, C_O, C_O, intermed["TEI"], optimize = True),
            "cvov": np.einsum('pP,qQ,rR,sS,pqrs->PQRS', C_C, C_V, C_O, C_V, intermed["TEI"], optimize = True),
            "covv": np.einsum('pP,qQ,rR,sS,pqrs->PQRS', C_C, C_O, C_V, C_V, intermed["TEI"], optimize = True),
            "cccc": np.einsum('pP,qQ,rR,sS,pqrs->PQRS', C_C, C_C, C_C, C_C, intermed["TEI"], optimize = True),
            "cccv": np.einsum('pP,qQ,rR,sS,pqrs->PQRS', C_C, C_C, C_C, C_V, intermed["TEI"], optimize = True),
            "cvvv": np.einsum('pP,qQ,rR,sS,pqrs->PQRS', C_C, C_V, C_V, C_V, intermed["TEI"], optimize = True),
            "cvoo": np.einsum('pP,qQ,rR,sS,pqrs->PQRS', C_C, C_V, C_O, C_O, intermed["TEI"], optimize = True),
            }
        H = {
            "cc": np.einsum('pP, qQ, pq -> PQ', C_C, C_C, intermed["OEI"], optimize = True),
            "co": np.einsum('pP, qQ, pq -> PQ', C_C, C_O, intermed["OEI"], optimize = True),
            "cv": np.einsum('pP, qQ, pq -> PQ', C_C, C_V, intermed["OEI"], optimize = True),
            "oo": np.einsum('pP, qQ, pq -> PQ', C_O, C_O, intermed["OEI"], optimize = True),
            "ov": np.einsum('pP, qQ, pq -> PQ', C_O, C_V, intermed["OEI"], optimize = True),
            "vv": np.einsum('pP, qQ, pq -> PQ', C_V, C_V, intermed["OEI"], optimize = True)
            }
        F = {
            "cc": H["cc"] + np.einsum('Pi Qi -> PQ', I["coco"], optimize = True) + np.einsum('iP iQ -> PQ', I["cccc"], optimize = True),
            "cv": H["cv"] - np.einsum('Pi iQ -> PQ', I["coov"], optimize = True) + np.einsum('iP iQ -> PQ', I["cccv"], optimize = True),
            "oo": H["oo"] + np.einsum('iP iQ -> PQ', I["oooo"], optimize = True) + np.einsum('iP iQ -> PQ', I["coco"], optimize = True),
            "ov": H["ov"] + np.einsum('iP iQ -> PQ', I["ooov"], optimize = True) + np.einsum('iP iQ -> PQ', I["cocv"], optimize = True),
            "vv": H["vv"] - np.einsum('Pi iQ -> PQ', I["voov"], optimize = True) + np.einsum('iP iQ -> PQ', I["cvcv"], optimize = True)
            }

    Fc = F["cc"].diagonal()
    Fo = F["oo"].diagonal()
    Fv = F["vv"].diagonal()
    D_ov = Fo.reshape(-1, 1) - Fv
    D_cv = Fc.reshape(-1, 1) - Fv
    # Use the diagonal MP2 hessian for core-occupied rotations.
    # However, off-diagonal elements can be large (confirmed numerically by code not in P4N)
    # so even this is suboptimal.
    D_co = np.einsum("C, Ii, iI -> CI", np.ones(num_cor), F["oo"], opdm_oo_corr)
    D_co -= np.einsum("C, II -> CI", Fc, opdm_oo_corr)
    D_co += 0.5 * np.einsum("C, Ijab, Ijab -> CI", np.ones(num_cor), t2, I["oovv"])
    D2 = Fo.reshape(-1, 1, 1, 1) + Fo.reshape(-1, 1, 1) - Fv.reshape(-1, 1) - Fv

    ### Construct reduced density matrices. Eq. 24 is always relevant.
    scf_opdm = np.eye(*F["oo"].shape) # Eq. 25
    opdm_oo_corr = - 0.5 * np.einsum("Ik ab,Jk ab->IJ", t2, t2, optimize=True) # Eq. 29
    opdm_oo = scf_opdm + opdm_oo_corr # OO block of eq. 23
    opdm_vv = + 0.5 * np.einsum("ij Ac,ij Bc->AB", t2, t2, optimize=True) # Eq. 30
    tpdm_oovv = t2.copy() # Eq. 34
    tpdm_vvvv = 0.5 * np.einsum("ijCD, ijAB -> ABCD", t2, t2, optimize=True) # Eq. 32
    tpdm_oooo = 0.5 * np.einsum("IJab, KLab -> IJKL", t2, t2, optimize=True) # Eq. 31
    # Eq. 24
    temp = np.einsum("p r, q s -> pqrs", scf_opdm, opdm_oo_corr, optimize=True)
    tpdm_oooo += temp + temp.transpose((1, 0, 3, 2)) - temp.transpose((1, 0, 2, 3)) - temp.transpose((0, 1, 3, 2))
    # Eq. 26
    temp = np.einsum("pr, qs -> pqrs", scf_opdm, scf_opdm, optimize=True)
    tpdm_oooo += temp - temp.transpose((0, 1, 3, 2))
    # Eq. 33
    tpdm_ovov = - np.einsum("IiBa, JiAa -> IAJB", t2, t2, optimize=True) + np.einsum("p r, q s -> pqrs", scf_opdm, opdm_vv, optimize=True)
    opdm_cc = np.eye(*F["cc"].shape)
    temp = np.einsum("p r, q s -> pqrs", opdm_cc, opdm_cc, optimize=True)
    tpdm_cccc = temp + temp.transpose((1, 0, 3, 2))
    tpdm_coco = np.einsum("pr, qs -> pqrs", opdm_cc, opdm_oo, optimize=True)
    tpdm_cvcv = np.einsum("pr, qs -> pqrs", opdm_cc, opdm_vv, optimize=True)

    # Expand out eq. 37.
    # A non-toy implementation will likely want to avoid explicitly constructing the VVVV TPDM
    # due to prohibitive memory requirements, and also separate out the product and non-product
    # terms of the TPDM, to use Fock matrices and see some core terms cancel.
    # We have invoked the factorization of any TPDM elements with core terms.
    r_ov = np.einsum("iA, Ii -> IA", H["ov"], opdm_oo)
    r_ov -= np.einsum("Ia, aA -> IA", H["ov"], opdm_vv)
    r_ov += 0.5 * np.einsum("jkiA, iIjk -> IA", I["ooov"], tpdm_oooo)
    r_ov += np.einsum("ckcA, Ik -> IA", I["cocv"], opdm_oo)
    r_ov -= 0.5 * np.einsum("iAab, Iiab -> IA", I["ovvv"], tpdm_oovv)
    r_ov += np.einsum("ibAa, Iaib -> IA", I["ovvv"], tpdm_ovov)
    r_ov -= 0.5 * np.einsum("Iabc, Aabc -> IA", I["ovvv"], tpdm_vvvv)
    r_ov -= 0.5 * np.einsum("ijIa, ijAa -> IA", I["ooov"], tpdm_oovv)
    r_ov -= np.einsum("iIja, iAja -> IA", I["ooov"], tpdm_ovov)
    r_ov -= np.einsum("cIca, Aa -> IA", I["cocv"], opdm_vv)
    t_ov += r_ov / D_ov

    r_co = H["co"].copy()
    r_co -= np.einsum("Ci, iI -> CI", H["co"], opdm_oo)
    r_co += np.einsum("cCcI -> CI", I["ccco"])
    r_co += np.einsum("CiIj, ij -> CI", I["cooo"], opdm_oo)
    r_co += np.einsum("CaIb, ab -> CI", I["cvov"], opdm_vv)
    r_co -= 0.5 * np.einsum("Cijk, Iijk -> CI", I["cooo"], tpdm_oooo)
    r_co -= 0.5 * np.einsum("Ciab, Iiab -> CI", I["covv"], tpdm_oovv)
    r_co -= np.einsum("Caib, Iaib -> CI", I["cvov"], tpdm_ovov)
    r_co -= np.einsum("cCci, I i -> CI", I["ccco"], opdm_oo)
    # On iteration 0, we're at SCF. Denominator and numerator are both zero.
    if i != 1:
        t_co += r_co / D_co

    r_cv = H["cv"].copy()
    r_cv -= np.einsum("Ca, aA -> CA", H["cv"], opdm_vv)
    r_cv += np.einsum("cCcA -> CA", I["cccv"])
    r_cv -= np.einsum("CijA, ij -> CA", I["coov"], opdm_oo)
    r_cv += np.einsum("CaAb, ab -> CA", I["cvvv"], opdm_vv)
    r_cv -= 0.5 * np.einsum("Cabc, Aabc -> CA", I["cvvv"], tpdm_vvvv)
    r_cv -= 0.5 * np.einsum("Cbij, ijAb -> CA", I["cvoo"], tpdm_oovv)
    r_cv += np.einsum("Cjib, jAib -> CA", I["coov"], tpdm_ovov)
    r_cv -= np.einsum("cCca, Aa -> CA", I["cccv"], opdm_vv)
    t_cv += r_cv / D_cv

    ### Construct T2 residual and step; Eq. 5
    # Two Electron Terms
    r2 = I["oovv"] + 0.5 * np.einsum("ABcd, IJcd -> IJAB", I["vvvv"], t2, optimize=True)
    r2 += 0.5 * np.einsum("klIJ, klAB -> IJAB", I["oooo"], t2, optimize=True)
    temp = np.einsum("AkIc, JkBc -> IJAB", I["voov"], t2, optimize=True)
    r2 += temp + temp.transpose((1, 0, 3, 2)) - temp.transpose((0, 1, 3, 2)) - temp.transpose((1, 0, 2, 3))
    # One Electron Terms. For canonical orbitals, this will reduce to -t2, after dividing by D
    temp = - np.einsum("Ii, iJAB -> IJAB", F["oo"], t2, optimize=True)
    r2 += temp - temp.transpose((1, 0, 2, 3))
    temp = + np.einsum("aA, IJaB -> IJAB", F["vv"], t2, optimize=True)
    r2 += temp - temp.transpose((0, 1, 3, 2))
    # Step
    t2 += r2 / D2

    ### DIIS and Print
    R = [r2, r_ov, r_cv, r_co]
    t2, t_ov, t_cv, t_co = dsd.diis(R, [t2, t_ov, t_cv, t_co])
    r_norm = np.sqrt(sum([np.linalg.norm(x) ** 2 for x in R]))
    Eref = SD_E(mol, H, I)
    print(Eref - Escf)
    Eolccd = 0.25 * np.sum(I["oovv"] * t2) + Eref - Escf
    print(f"{i:3d} E={Eolccd:3.10f} R = {r_norm:0.8f}")
    if r_norm < float(f"1e-{target_convergence}"):
        break
else:
    raise Exception("Equations did not converge.")

if compare_psi4:
   wfn = psi4.energy("olccd", return_wfn=True)[1]
   psi4.compare_values(psi4.variable("CURRENT CORRELATION ENERGY"), Eolccd, target_convergence, "OLCCD Energy")
