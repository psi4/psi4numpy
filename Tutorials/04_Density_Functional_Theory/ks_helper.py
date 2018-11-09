#! A simple Psi4 input script to compute a SCF reference using Psi4's libJK

import time
import numpy as np
import psi4

from pkg_resources import parse_version
if parse_version(psi4.__version__) >= parse_version('1.3a1'):
    build_superfunctional = psi4.driver.dft.build_superfunctional
else:
    build_superfunctional = psi4.driver.dft_funcs.build_superfunctional
    
# Diagonalize routine
def build_orbitals(diag, A, ndocc):
    Fp = psi4.core.Matrix.triplet(A, diag, A, True, False, True)

    nbf = A.shape[0]
    Cp = psi4.core.Matrix(nbf, nbf)
    eigvecs = psi4.core.Vector(nbf)
    Fp.diagonalize(Cp, eigvecs, psi4.core.DiagonalizeOrder.Ascending)

    C = psi4.core.Matrix.doublet(A, Cp, False, False)

    Cocc = psi4.core.Matrix(nbf, ndocc)
    Cocc.np[:] = C.np[:, :ndocc]

    D = psi4.core.Matrix.doublet(Cocc, Cocc, False, True)
    return C, Cocc, D, eigvecs

def ks_solver(alias, mol, options, V_builder, jk_type="DF", output="output.dat", restricted=True):

    # Build our molecule
    mol = mol.clone()
    mol.reset_point_group('c1')
    mol.fix_orientation(True)
    mol.fix_com(True)
    mol.update_geometry()

    # Set options
    psi4.set_output_file(output)

    psi4.core.prepare_options_for_module("SCF")
    psi4.set_options(options)
    psi4.core.set_global_option("SCF_TYPE", jk_type)

    maxiter = 20
    E_conv = psi4.core.get_option("SCF", "E_CONVERGENCE") 
    D_conv = psi4.core.get_option("SCF", "D_CONVERGENCE")
    
    # Integral generation from Psi4's MintsHelper
    wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option("BASIS"))
    mints = psi4.core.MintsHelper(wfn.basisset())
    S = mints.ao_overlap()

    # Build the V Potential
    sup = build_superfunctional(alias, restricted)[0]
    sup.set_deriv(2)
    sup.allocate()
    
    vname = "RV"
    if not restricted:
        vname = "UV"
    Vpot = psi4.core.VBase.build(wfn.basisset(), sup, vname)
    Vpot.initialize()
    
    # Get nbf and ndocc for closed shell molecules
    nbf = wfn.nso()
    ndocc = wfn.nalpha()
    if wfn.nalpha() != wfn.nbeta():
        raise PsiException("Only valid for RHF wavefunctions!")
    
    print('\nNumber of occupied orbitals: %d' % ndocc)
    print('Number of basis functions:   %d' % nbf)
    
    # Build H_core
    V = mints.ao_potential()
    T = mints.ao_kinetic()
    H = T.clone()
    H.add(V)
    
    # Orthogonalizer A = S^(-1/2)
    A = mints.ao_overlap()
    A.power(-0.5, 1.e-14)
    
    # Build core orbitals
    C, Cocc, D, eigs = build_orbitals(H, A, ndocc)
    
    # Setup data for DIIS
    t = time.time()
    E = 0.0
    Enuc = mol.nuclear_repulsion_energy()
    Eold = 0.0
    
    # Initialize the JK object
    jk = psi4.core.JK.build(wfn.basisset())
    jk.set_memory(int(1.25e8))  # 1GB
    jk.initialize()
    jk.print_header()
    
    diis_obj = psi4.p4util.solvers.DIIS(max_vec=3, removal_policy="largest")
    
    print('\nTotal time taken for setup: %.3f seconds' % (time.time() - t))
    
    print('\nStarting SCF iterations:')
    t = time.time()
   
    print("\n    Iter            Energy             XC E         Delta E        D RMS\n")
    for SCF_ITER in range(1, maxiter + 1):
    
        # Compute JK
        jk.C_left_add(Cocc)
        jk.compute()
        jk.C_clear()
    
        # Build Fock matrix
        F = H.clone()
        F.axpy(2.0, jk.J()[0])
        F.axpy(-Vpot.functional().x_alpha(), jk.K()[0])

        # Build V
        ks_e = 0.0

        Vpot.set_D([D])
        Vpot.properties()[0].set_pointers(D)
        V = V_builder(D, Vpot)
        if V is None:
            ks_e = 0.0
        else:
            ks_e, V = V
            V = psi4.core.Matrix.from_array(V)
    
        F.axpy(1.0, V)

        # DIIS error build and update
        diis_e = psi4.core.Matrix.triplet(F, D, S, False, False, False)
        diis_e.subtract(psi4.core.Matrix.triplet(S, D, F, False, False, False))
        diis_e = psi4.core.Matrix.triplet(A, diis_e, A, False, False, False)
    
        diis_obj.add(F, diis_e)
    
        dRMS = diis_e.rms()

        # SCF energy and update
        SCF_E  = 2.0 * H.vector_dot(D)
        SCF_E += 2.0 * jk.J()[0].vector_dot(D)
        SCF_E -= Vpot.functional().x_alpha() * jk.K()[0].vector_dot(D)
        SCF_E += ks_e
        SCF_E += Enuc
    
        print('SCF Iter%3d: % 18.14f   % 11.7f   % 1.5E   %1.5E'
              % (SCF_ITER, SCF_E, ks_e, (SCF_E - Eold), dRMS))
        if (abs(SCF_E - Eold) < E_conv) and (dRMS < D_conv):
            break
    
        Eold = SCF_E
    
        # DIIS extrapolate
        F = diis_obj.extrapolate()
    
        # Diagonalize Fock matrix
        C, Cocc, D, eigs = build_orbitals(F, A, ndocc)
    
        if SCF_ITER == maxiter:
            raise Exception("Maximum number of SCF cycles exceeded.")
    
    print('\nTotal time for SCF iterations: %.3f seconds ' % (time.time() - t))
    
    print('\nFinal SCF energy: %.8f hartree' % SCF_E)

    data = {}
    data["Da"] = D
    data["Ca"] = C
    data["eigenvalues"] = eigs
    return(SCF_E, data)
