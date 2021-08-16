"""
Helper function for CQED_RHF

"""

__authors__   = ["Jon McTague", "Jonathan Foley"]
__credits__   = ["Jon McTague", "Jonathan Foley"]

__copyright_amp__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__   = "BSD-3-Clause"
__date__      = "2021-01-15"

# ==> Import Psi4, NumPy, & SciPy <==
import psi4
import numpy as np
import scipy.linalg as la
import time

def cis(molecule_string, psi4_options_dict):
    """ Computes the CIS energy and total dipole moment for select states

        Arguments
        ---------
        molecule_string : string
            specifies the molecular geometry

        psi4_options_dict : dictionary
            specifies psi4 options

        states : array of ints
            specifies CIS statees for which the energies and total dipole moments should be returned

        Returns
        -------

        cis_energy : array of floats
            cis energy for states

        cis_total_dipole_moments : array of 1x3 vectors
            total dipole moments for states

    """
    # hard-code this for now... think of optional keywords later!
    # this is the total dipole moment
    cis_dipole = np.zeros((3,2))
    # this is the transition dipole moment between g->e1
    tdm = np.zeros(3)
    
    # define geometry using the molecule_string
    mol = psi4.geometry(molecule_string)
    # define options for the calculation
    psi4.set_options(psi4_options_dict)
    # run psi4 to get ordinary scf energy and wavefunction object
    scf_e, wfn = psi4.energy('scf', return_wfn=True)
    
    # ==> Nuclear Repulsion Energy <==
    E_nuc = mol.nuclear_repulsion_energy()
    nmo = wfn.nmo()

    # Create instance of MintsHelper class
    mints = psi4.core.MintsHelper(wfn.basisset())
    
    # Grab data from wavfunction
    
    # number of doubly occupied orbitals
    ndocc   = wfn.nalpha()
    
    # total number of orbitals
    nmo     = wfn.nmo()
    
    # number of virtual orbitals
    nvirt   = nmo - ndocc
    
    # grab all transformation vectors and store to a numpy array!
    C = np.asarray(wfn.Ca())
    
    # occupied orbitals:
    Co = wfn.Ca_subset("AO", "OCC")
    
    # virtual orbitals:
    Cv = wfn.Ca_subset("AO", "VIR")
    
    # grab all transformation vectors and store to a numpy array!
    C = np.asarray(wfn.Ca())
    
    # orbital energies
    eps     = np.asarray(wfn.epsilon_a())
    
    # ==> Nuclear Repulsion Energy <==
    E_nuc = mol.nuclear_repulsion_energy()
    
    print("\nNumber of occupied orbitals: %d" % ndocc)
    
    # 2 electron integrals in ao basis
    #I = np.asarray(mints.ao_eri())

    # 2 electron integrals in mo basis
    ovov = np.asarray(mints.mo_eri(Co, Cv, Co, Cv))
    
    # build the (oo|vv) integrals:
    oovv = np.asarray(mints.mo_eri(Co, Co, Cv, Cv))

    # strip out occupied orbital energies, eps_o spans 0..ndocc-1
    eps_o = eps[:ndocc]
    
    # strip out virtual orbital energies, eps_v spans 0..nvirt-1
    eps_v = eps[ndocc:]

    # create Hamiltonian
    HCIS = np.zeros((ndocc * nvirt, ndocc * nvirt))

    for i in range(0, ndocc):
        for a in range(0, nvirt):
            ia = i*nvirt + a
            for j in range(0, ndocc):
                for b in range(0, nvirt):
                    jb = j*nvirt + b
                    term1 = eps_v[a] - eps_o[i]
                    term2 = 2 * ovov[i, a, j, b] - oovv[i,j,a,b]
                    
                    if (i==j) and (a == b):
                        HCIS[ia, jb] = term1 + term2
                        
                    else:
                        HCIS[ia, jb] = term2

    ECIS, CCIS = np.linalg.eigh(HCIS)

    # get ready to compute diple moments
    # nuclear dipole
    mu_nuc_x = mol.nuclear_dipole()[0]
    mu_nuc_y = mol.nuclear_dipole()[1]
    mu_nuc_z = mol.nuclear_dipole()[2]
    
    # dipole arrays in AO basis
    mu_ao_x = np.asarray(mints.ao_dipole()[0])
    mu_ao_y = np.asarray(mints.ao_dipole()[1])
    mu_ao_z = np.asarray(mints.ao_dipole()[2])

    # transform dipole array to canonical MO basis
    mu_cmo_x = np.dot(C.T, mu_ao_x).dot(C)
    mu_cmo_y = np.dot(C.T, mu_ao_y).dot(C)
    mu_cmo_z = np.dot(C.T, mu_ao_z).dot(C)
    
    for i in range(0, ndocc):
        # double because this is only alpha terms!
        cis_dipole[0,0] += 2 * mu_cmo_x[i, i]
        cis_dipole[1,0] += 2 * mu_cmo_y[i, i]
        cis_dipole[2,0] += 2 * mu_cmo_z[i, i]

    # need to add the nuclear term to the expectation values above which
    # only included the electronic term!
    # go ahead and add to excited-state term as well!
    cis_dipole[0,0] += mu_nuc_x 
    cis_dipole[1,0] += mu_nuc_y
    cis_dipole[2,0] += mu_nuc_z

    cis_dipole[0,1] += mu_nuc_x
    cis_dipole[1,1] += mu_nuc_y
    cis_dipole[2,1] += mu_nuc_z

    # first excited state total dipole moment contains contribution 
    # sum_{i,j,a} c_i^a c_j^a \mu_aa
    # and transition dipole moment contains contribution 
    # sum_ia c_ia \mu_ia
    for i in range(0, ndocc):
        for a in range(0, nvirt):
            ia = i*nvirt + a
            # transition dipole moment part - 
            # NOTE: 1/sqrt(2) arises bc of spin adaptation
            tdm[0] += 2 * CCIS[ia,0] * mu_cmo_x[i,a+ndocc] / np.sqrt(2)
            tdm[1] += 2 * CCIS[ia,0] * mu_cmo_y[i,a+ndocc] / np.sqrt(2)
            tdm[2] += 2 * CCIS[ia,0] * mu_cmo_z[i,a+ndocc] / np.sqrt(2)

            for j in range(0, ndocc):
                ja = j*nvirt + a
                # total dipole moment part... 1/2 arises bc of spin adaptation, cancelling 
                # factor of 2 that occurs in the ground state!
                cis_dipole[0,1] += CCIS[ia,0] * CCIS[ja,0] * mu_cmo_x[a+ndocc,a+ndocc] 
                cis_dipole[1,1] += CCIS[ia,0] * CCIS[ja,0] * mu_cmo_y[a+ndocc,a+ndocc]
                cis_dipole[2,1] += CCIS[ia,0] * CCIS[ja,0] * mu_cmo_z[a+ndocc,a+ndocc]

    # -sum_{a,b,i} c_i^a c_i^b \mu_ii
    for i in range(0, ndocc):
        for a in range(0, nvirt):
            ia = i*nvirt + a
            for b in range(0, nvirt):
                ib = i*nvirt + b
                cis_dipole[0,1] -= CCIS[ia,0] * CCIS[ib,0] * mu_cmo_x[i,i]
                cis_dipole[1,1] -= CCIS[ia,0] * CCIS[ib,0] * mu_cmo_y[i,i]
                cis_dipole[2,1] -= CCIS[ia,0] * CCIS[ib,0] * mu_cmo_z[i,i]

    # return first excited state energy
    # ground and first excited state total dipole moment
    # and g->e1 transition dipole moment
    return ECIS, cis_dipole, tdm
