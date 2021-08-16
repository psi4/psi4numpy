"""
Helper function for CQED-CIS in the coherent state basis

References:
    Equations and algorithms from 
    [Haugland:2020:041043], [DePrince:2021:094112], and [McTague:2021:] 

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
from helper_cqed_rhf import cqed_rhf

def cs_cqed_cis(lambda_vector, omega_val, molecule_string, psi4_options_dict):
    """ Computes the QED-RHF energy and density 

        Arguments
        ---------
        lambda_vector : 1 x 3 array of floats
            the electric field vector, see e.g. Eq. (1) in [DePrince:2021:094112]
            and (15) in [Haugland:2020:041043]

        omega_val : complex float
            the complex energy associated with the photon, see Eq. (3) in [McTague:2021:]
        
        molecule_string : string
            specifies the molecular geometry

        psi4_options_dict : dictionary
            specifies the psi4 options to be used in running requisite psi4 calculations 

        Returns
        -------
        cqed_cis_dictionary : dictionary
            Contains important quantities from the cqed_rhf calculation, with keys including:
                'RHF ENERGY' -> result of canonical RHF calculation using psi4 defined by molecule_string and psi4_options_dict
                'CQED-RHF ENERGY' -> result of CQED-RHF calculation, see Eq. (13) of [McTague:2021:] 
                'CQED-CIS ENERGY' -> numpy array of complex floats comprising energy eigenvalues of CQED-CIS Hamiltonian
                'CQED-CIS L VECTORS' -> numpy array of complex floats comprising the left eigenvectors of CQED-CIS Hamiltonian  

        Example
        -------
        >>> cqed_cis_dictionary = cs_cqed_cis([0., 0., 1e-2], 0.2-0.001j, '''\nMg\nH 1 1.7\nsymmetry c1\n1 1\n''', psi4_options_dictionary)
        
    """
    
    # define geometry using the molecule_string
    mol = psi4.geometry(molecule_string)
    # define options for the calculation
    psi4.set_options(psi4_options_dict)
    # run psi4 to get ordinary scf energy and wavefunction object
    #scf_e, wfn = psi4.energy('scf', return_wfn=True)

    # run cqed_rhf method
    cqed_rhf_dict = cqed_rhf(lambda_vector, molecule_string, psi4_options_dict)
    
    # grab necessary quantities from cqed_rhf_dict
    scf_e  = cqed_rhf_dict['RHF ENERGY']
    cqed_scf_e = cqed_rhf_dict['CQED-RHF ENERGY']
    wfn = cqed_rhf_dict['PSI4 WFN']
    C = cqed_rhf_dict['CQED-RHF C']
    eps = cqed_rhf_dict['CQED-RHF EPS']
    cqed_rhf_dipole_moment = cqed_rhf_dict['CQED-RHF DIPOLE MOMENT']

    # Create instance of MintsHelper class
    mints = psi4.core.MintsHelper(wfn.basisset())
    
    # Grab data from wavfunction
    
    # number of doubly occupied orbitals
    ndocc   = wfn.nalpha()
    
    # total number of orbitals
    nmo     = wfn.nmo()
    
    # number of virtual orbitals
    nvirt   = nmo - ndocc

    # need to update the Co and Cv core matrix objects so we can
    # utlize psi4s fast integral transformation!

    # collect rhf wfn object as dictionary
    wfn_dict = psi4.core.Wavefunction.to_file(wfn)
    
    # update wfn_dict with orbitals from CQED-RHF 
    wfn_dict['matrix']['Ca'] = C
    wfn_dict['matrix']['Cb'] = C
    # update wfn object
    wfn = psi4.core.Wavefunction.from_file(wfn_dict) 

    # occupied orbitals as psi4 objects but they correspond to CQED-RHF orbitals
    Co = wfn.Ca_subset("AO", "OCC")
    
    # virtual orbitals same way
    Cv = wfn.Ca_subset("AO", "VIR")
    
    # 2 electron integrals in CQED-RHF basis
    ovov = np.asarray(mints.mo_eri(Co, Cv, Co, Cv))
    
    # build the (oo|vv) integrals:
    oovv = np.asarray(mints.mo_eri(Co, Co, Cv, Cv))

    # strip out occupied orbital energies, eps_o spans 0..ndocc-1
    eps_o = eps[:ndocc]
    
    # strip out virtual orbital energies, eps_v spans 0..nvirt-1
    eps_v = eps[ndocc:]
    
    # Extra terms for Pauli-Fierz Hamiltonian
    # nuclear dipole
    mu_nuc_x = mol.nuclear_dipole()[0]
    mu_nuc_y = mol.nuclear_dipole()[1]
    mu_nuc_z = mol.nuclear_dipole()[2]

    # l \cdot \mu_nuc for d_c 
    l_dot_mu_nuc = lambda_vector[0] * mu_nuc_x
    l_dot_mu_nuc += lambda_vector[1] * mu_nuc_y
    l_dot_mu_nuc += lambda_vector[2] * mu_nuc_z
    
    # dipole arrays in AO basis
    mu_ao_x = np.asarray(mints.ao_dipole()[0])
    mu_ao_y = np.asarray(mints.ao_dipole()[1])
    mu_ao_z = np.asarray(mints.ao_dipole()[2])
    
    # transform dipole array to CQED-RHF basis
    mu_cmo_x = np.dot(C.T, mu_ao_x).dot(C)
    mu_cmo_y = np.dot(C.T, mu_ao_y).dot(C)
    mu_cmo_z = np.dot(C.T, mu_ao_z).dot(C)

    # \lambda \cdot < \mu > 
    # e.g. line 6 of Eq. (18) in [McTague:2021:]
    l_dot_mu_exp = 0.
    for i in range(0,3):
        l_dot_mu_exp += lambda_vector[i] * cqed_rhf_dipole_moment[i]

    # \lambda \cdot \mu_{el}
    # e.g. line 4 Eq. (18) in [McTague:2021:]
    l_dot_mu_el =  lambda_vector[0] * mu_cmo_x 
    l_dot_mu_el += lambda_vector[1] * mu_cmo_y
    l_dot_mu_el += lambda_vector[2] * mu_cmo_z
    
    # dipole constants to add to E_CQED_CIS, 
    #  0.5 * (\lambda \cdot \mu_{nuc})** 2 
    #      - (\lambda \cdot <\mu> ) ( \lambda \cdot \mu_{nuc})
    # +0.5 * (\lambda \cdot <\mu>) ** 2
    # Eq. (14) of [McTague:2021:]
    d_c = 0.5 * l_dot_mu_nuc ** 2 - l_dot_mu_nuc * l_dot_mu_exp + 0.5 * l_dot_mu_exp ** 2

    # check to see if d_c what we have from CQED-RHF calculation
    assert np.isclose(d_c, cqed_rhf_dict['DIPOLE ENERGY'])

    # create Hamiltonian for elements H[ias, jbt]
    Htot = np.zeros((ndocc * nvirt * 2 + 2, ndocc * nvirt * 2 + 2), dtype=complex)
    Hep = np.zeros((ndocc * nvirt * 2 + 2, ndocc * nvirt * 2 + 2), dtype=complex)
    H1e = np.zeros((ndocc * nvirt * 2 + 2, ndocc * nvirt * 2 + 2), dtype=complex)
    H2e = np.zeros((ndocc * nvirt * 2 + 2, ndocc * nvirt * 2 + 2), dtype=complex)
    H2edp = np.zeros((ndocc * nvirt * 2 + 2, ndocc * nvirt * 2 + 2), dtype=complex)
    Hp = np.zeros((ndocc * nvirt * 2 + 2, ndocc * nvirt * 2 + 2), dtype=complex)

    # elements corresponding to <s|<\Phi_0 | H | \Phi_0>|t>
    # Eq. (16) of [McTague:2021:]
    Hp[0,0] = 0.
    Hp[1,1] = omega_val 

    # elements corresponding to <s|<\Phi_0 | H | \Phi_i^a>|t>
    # Eq. (17) of [McTague:2021:]
    for s in range(0,2):
        for i in range(0,ndocc):
            for a in range(0,nvirt):
                A = a + ndocc
                for t in range(0,2):
                    iat = 2*(i*nvirt + a) + t + 2
                    Hep[s,iat] = -np.sqrt(omega_val) * np.sqrt(t+1) * l_dot_mu_el[i,A] * (s==t+1) 
                    Hep[s,iat] -= np.sqrt(omega_val) * np.sqrt(t) * l_dot_mu_el[i,A] * (s==t-1) 
                    Hep[iat,s] = -np.sqrt(omega_val) * np.sqrt(s+1) * l_dot_mu_el[i,A] * (s+1==t) 
                    Hep[iat,s] -= np.sqrt(omega_val) * np.sqrt(s) * l_dot_mu_el[i,A] * (s-1==t) 
    

    # elements corresponding to <s|<\Phi_i^a| H | \Phi_j^b|t>
    # Eq. (18) of [McTague:2021:]
    for i in range(0, ndocc):
        for a in range(0, nvirt):
            A = a+ndocc
            for s in range(0,2):
                ias = 2*(i*nvirt + a) + s + 2
                
                for j in range(0, ndocc):
                    for b in range(0, nvirt):
                        B = b+ndocc
                        for t in range(0,2):
                            jbt = 2*(j*nvirt + b) + t + 2
                            # ERIs
                            H2e[ias,jbt] =  (2.0 * ovov[i, a, j, b] - oovv[i, j, a, b]) * (s==t)
                            # 2-electron dipole terms
                            # ordinary
                            H2edp[ias,jbt] += (2.0 * l_dot_mu_el[i,A] * l_dot_mu_el[j,B]) * (s==t)
                            # exchange
                            H2edp[ias,jbt] -= l_dot_mu_el[i,j] * l_dot_mu_el[A,B] * (s==t)
                            # orbital energies from CQED-RHF
                            H1e[ias,jbt] += eps_v[a] * (s==t) * (a==b) * (i==j)
                            H1e[ias,jbt] -= eps_o[i] * (s==t) * (a==b) * (i==j)
                            # photonic and dipole energy term
                            Hp[ias,jbt] += (omega_val * t) * (s==t) * (i==j) * (a==b)
                            # bilinear coupling - off-diagonals first
                            Hep[ias,jbt] += np.sqrt(t+1) * np.sqrt(omega_val/2) * l_dot_mu_el[i,j] * (s==t+1) * (a==b) 
                            Hep[ias,jbt] += np.sqrt(t) * np.sqrt(omega_val/2) * l_dot_mu_el[i,j] * (s==t-1) * (a==b) 
                            Hep[ias,jbt] -= np.sqrt(t+1) * np.sqrt(omega_val/2) * l_dot_mu_el[A,B] * (s==t+1) * (i==j)
                            Hep[ias,jbt] -= np.sqrt(t) * np.sqrt(omega_val/2) * l_dot_mu_el[A,B] * (s==t-1) * (i==j)
                            # now handle diagonal in electronic term
                            if (a==b and i==j and s==t+1):
                                # l dot <mu> term
                                Hep[ias,jbt] += np.sqrt(t+1) * np.sqrt(omega_val/2) * l_dot_mu_exp
                                # l dot mu terms
                                for k in range(0,ndocc):
                                    # sum over occupied indices
                                    Hep[ias,jbt] -= np.sqrt(t+1) * np.sqrt(omega_val/2) * l_dot_mu_el[k,k]


                            # now handle diagonal in electronic term
                            if (a==b and i==j and s==t-1):
                                # l dot <mu> term
                                Hep[ias,jbt] += np.sqrt(t) * np.sqrt(omega_val/2) * l_dot_mu_exp
                                # l dot mu terms
                                for k in range(0,ndocc):
                                    # sum over occupied indices
                                    Hep[ias,jbt] -= np.sqrt(t) * np.sqrt(omega_val/2) * l_dot_mu_el[k,k]
    # Form Htot from sum of all terms
    Htot = Hp + Hep + H1e + H2e + H2edp
    # now diagonalize H
    # use eigh if Hermitian
    if np.isclose(np.imag(omega_val),0,1e-6):
        ECIS, CCIS = np.linalg.eigh(Htot)
    # use eig if not-Hermitian.  Note that
    # numpy eig just returns the left eigenvectors
    # and does not sort the eigenvalues
    else:
        ECIS, CCIS = np.linalg.eig(Htot)
        idx = ECIS.argsort()
        ECIS = ECIS[idx]
        CCIS = CCIS[:,idx]

    # ENERGY DECOMPOSITION ANALYSIS
    C_LP_star = np.conj(CCIS[:,1].T)
    C_UP_star = np.conj(CCIS[:,2])
    C_LP = CCIS[:,1]
    C_UP = CCIS[:,2]

    Htot_on_LP = np.dot(Htot, C_LP)
    Hp_on_LP = np.dot(Hp, C_LP)
    Hep_on_LP = np.dot(Hep, C_LP)
    H1e_on_LP = np.dot(H1e, C_LP)
    H2e_on_LP = np.dot(H2e, C_LP)
    H2edp_on_LP = np.dot(H2edp, C_LP)

    Htot_on_UP = np.dot(Htot, C_UP)
    Hp_on_UP = np.dot(Hp, C_UP)
    Hep_on_UP = np.dot(Hep, C_UP)
    H1e_on_UP = np.dot(H1e, C_UP)
    H2e_on_UP = np.dot(H2e, C_UP)
    H2edp_on_UP = np.dot(H2edp, C_UP)

    Etot_LP = np.dot(C_LP_star, Htot_on_LP)
    Ep_LP = np.dot(C_LP_star, Hp_on_LP)
    Eep_LP = np.dot(C_LP_star, Hep_on_LP)
    E1e_LP = np.dot(C_LP_star, H1e_on_LP)
    E2e_LP = np.dot(C_LP_star, H2e_on_LP)
    E2edp_LP = np.dot(C_LP_star, H2edp_on_LP)

    Etot_UP = np.dot(C_UP_star, Htot_on_UP)
    Ep_UP = np.dot(C_UP_star, Hp_on_UP)
    Eep_UP = np.dot(C_UP_star, Hep_on_UP)
    E1e_UP = np.dot(C_UP_star, H1e_on_UP)
    E2e_UP = np.dot(C_UP_star, H2e_on_UP)
    E2edp_UP = np.dot(C_UP_star, H2edp_on_UP)

    assert(np.isclose(Etot_LP, ECIS[1]))
    assert(np.isclose(Etot_UP, ECIS[2]))

    cqed_cis_dict = {
                'RHF ENERGY' : scf_e,
                'CQED-RHF ENERGY' : cqed_scf_e,
                'CQED-CIS ENERGY' : ECIS,
                'CQED-CIS L VECTORS' : CCIS,
                'LP PHOTONIC ENERGY' : Ep_LP,
                'UP PHOTONIC ENERGY' : Ep_UP,
                'LP ELECTRON-PHOTON ENERGY' : Eep_LP,
                'UP ELECTRON-PHOTON ENERGY' : Eep_UP,
                'LP 1E ENERGY' : E1e_LP,
                'UP 1E ENERGY' : E1e_UP,
                'LP 2E ENERGY' : E2e_LP,
                'UP 2E ENERGY' : E2e_UP,
                'LP 2E DIPOLE ENERGY' : E2edp_LP,
                'UP 2E DIPOLE ENERGY' : E2edp_UP
    }

    return cqed_cis_dict


def get_nto(cis_vec, mo_occ, mo_virt):

    """ compute the natural transition orbitals from
        the cqed_cis wavefunction for the LP and UP states
        
        References
        ----------
        
            Original reference: Martin, R. L., JCP, 118, 4775-4777
        
            See also: Equations (1) - (7) of https://comp.chem.umn.edu/openmolcas/200303_NTOv12.pdf

            Python code from pyscf/tdscf module, see line 216 of https://github.com/pyscf/pyscf/blob/master/pyscf/tdscf/rhf.py
        
    """
    # we will get T amplitudes for LP and UP states
    # and note that there are essentially 2 sets of T amplitudes for each i->a excitation:
    # i0->a0 and i0->a1
    # we will get both
    ndocc = len(mo_occ[0,:])
    nvirt = len(mo_virt[0,:])
    t_lp = np.zeros((ndocc, nvirt, 2))
    t_up = np.zeros((ndocc, nvirt, 2))

    for i in range(0, ndocc):
        for a in range(0, nvirt):
            iaz = 2*(i*nvirt + a) + 0 + 2
            iao = 2*(i*nvirt + a) + 1 + 2
            # i0->a0
            t_lp[i,a,0] = cis_vec[iaz,1] 
            t_up[i,a,0] = cis_vec[iaz,2]
            # i0->a1 
            t_lp[i,a,1] = cis_vec[iao,1] 
            t_up[i,a,1] = cis_vec[iao,2]


    # perform svd
    # LP i0->a0 amplitudes
    u_lp_0, w_lp_0, vT_lp_0 = np.linalg.svd(t_lp[:,:,0])
    # UP i0->a0 amplitudes
    u_up_0, w_up_0, vT_up_0 = np.linalg.svd(t_up[:,:,0])
    # LP i0->a1 amplitudes
    u_lp_1, w_lp_1, vT_lp_1 = np.linalg.svd(t_lp[:,:,1])
    # UP i0->a1 amplitudes
    u_up_1, w_up_1, vT_up_1 = np.linalg.svd(t_up[:,:,1])

    # get v vectors
    v_lp_0 = vT_lp_0.conj().T
    v_lp_1 = vT_lp_1.conj().T
    v_up_0 = vT_up_0.conj().T
    v_up_1 = vT_up_1.conj().T
    
    # reorder these arrays
    # lp_0
    idx = np.argmax(abs(u_lp_0.real), axis=0)
    u_lp_0[:,u_lp_0[idx,np.arange(ndocc)].real<0] *= -1
    idx = np.argmax(abs(v_lp_0.real), axis=0)
    v_lp_0[:,v_lp_0[idx,np.arange(nvirt)].real<0] *= -1

    # lp_1
    idx = np.argmax(abs(u_lp_1.real), axis=0)
    u_lp_1[:,u_lp_1[idx,np.arange(ndocc)].real<0] *= -1
    idx = np.argmax(abs(v_lp_1.real), axis=0)
    v_lp_1[:,v_lp_1[idx,np.arange(nvirt)].real<0] *= -1

    # up_0
    idx = np.argmax(abs(u_up_0.real), axis=0)
    u_up_0[:,u_up_0[idx,np.arange(ndocc)].real<0] *= -1
    idx = np.argmax(abs(v_up_0.real), axis=0)
    v_up_0[:,v_up_0[idx,np.arange(nvirt)].real<0] *= -1

    # up_1
    idx = np.argmax(abs(u_up_1.real), axis=0)
    u_up_1[:,u_up_1[idx,np.arange(ndocc)].real<0] *= -1
    idx = np.argmax(abs(v_up_1.real), axis=0)
    v_up_1[:,v_up_1[idx,np.arange(nvirt)].real<0] *= -1

    # assemble full NTOs for each case
    # lp0
    occ_nto = np.dot(mo_occ, u_lp_0)
    vir_nto = np.dot(mo_virt, v_lp_0)
    nto_lp_0 = np.hstack((occ_nto, vir_nto))
    # lp1
    occ_nto = np.dot(mo_occ, u_lp_1)
    vir_nto = np.dot(mo_virt, v_lp_1)
    nto_lp_1 = np.hstack((occ_nto, vir_nto))
    # up0
    occ_nto = np.dot(mo_occ, u_up_0)
    vir_nto = np.dot(mo_virt, v_up_0)
    nto_up_0 = np.hstack((occ_nto, vir_nto))
    # up1
    occ_nto = np.dot(mo_occ, u_up_1)
    vir_nto = np.dot(mo_virt, v_up_1)
    nto_up_1 = np.hstack((occ_nto, vir_nto))

    # square singular values to get weights for each set of transitions
    w_lp_0 *= w_lp_0
    w_lp_1 *= w_lp_1
    w_up_0 *= w_up_0
    w_up_1 *= w_up_1

    # store all arrays in dictionary
    nto_dict = {
        'NTO LP0' : nto_lp_0,
        'NTO LP1' : nto_lp_1,
        'NTO UP0' : nto_up_0,
        'NTO UP1' : nto_up_1,
        'weights LP0' : w_lp_0,
        'weights LP1' : w_lp_1,
        'weights UP0' : w_up_0,
        'weights UP1' : w_up_1
    }

    return nto_dict
