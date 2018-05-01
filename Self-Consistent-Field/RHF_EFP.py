"""
Reference implementation of RHF/EFP using libefp through PylibEFP.

Requirements:
NumPy
PylibEFP >=0.1
libEFP >=1.5b1
Psi4 >=1.2a1.dev507 (c. late Aug 2017)

References:
SCF in Python from @dgasmith's most excellent Self-Consistent-Field/RHF.py .
SCF/EFP in Psi4 by @andysim, @edeprince3, @ilyak, @loriab
libefp from [Kaliman:2013:2284]

"""
from __future__ import division
from __future__ import print_function

__authors__   = "Lori A. Burns"
__credits__   = ["Andrew C. Simmonett", "A. Eugene DePrince III", "Ilya A. Kaliman", "Lori A. Burns", "Daniel G. A. Smith"]

__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__   = "BSD-3-Clause"
__date__      = "2017-08-28"

import time
import numpy as np
np.set_printoptions(precision=5, linewidth=200, suppress=True)
import psi4
import pylibefp

import os


# Memory for Psi4 in GB
psi4.set_memory('500 MB')
psi4.core.set_output_file("output.dat", False)

# Memory for numpy in GB
numpy_memory = 2

def set_qm_atoms(mol, efpobj):
    """Provides list of coordinates of quantum mechanical atoms from
    psi4.core.Molecule `mol` to pylibefp.core.efp() `efpobj`.

    """
    ptc = []
    coords = []
    for iat in range(mol.natom()):
        ptc.append(mol.charge(iat))
        coords.append(mol.x(iat))
        coords.append(mol.y(iat))
        coords.append(mol.z(iat))

    efpobj.set_point_charges(ptc, coords)


def modify_Fock_permanent(mol, nbf, efpobj):
    """Computes array of the EFP contribution to the potential felt by
    QM atoms, due to permanent EFP moments, for a SCF procedure.

    Requires psi4.core.Molecule `mol`, number of basis functions `nbf`,
    and pylibefp.core.efp() `efpobj`.

    """
    # get composition counts from libefp
    n_fr = efpobj.get_frag_count()
    natoms = efpobj.get_frag_atom_count()

    # get multipoles count, pos'n, values from libefp
    #   charge + dipoles + quadrupoles + octupoles = 20
    n_mp = efpobj.get_multipole_count()
    xyz_mp = np.asarray(efpobj.get_multipole_coordinates()).reshape(n_mp, 3)
    val_mp = np.asarray(efpobj.get_multipole_values()).reshape(n_mp, 20)

    #                    0  X  Y  Z  XX   YY   ZZ   XY   XZ   YZ
    prefacs = np.array([ 1, 1, 1, 1, 1/3, 1/3, 1/3, 2/3, 2/3, 2/3,
        1/15, 1/15, 1/15, 3/15, 3/15, 3/15, 3/15, 3/15, 3/15, 6/15])
    #   XXX   YYY   ZZZ   XXY   XXZ   XYY   YYZ   XZZ   YZZ   XYZ

    # EFP permanent moment contribution to the Fock Matrix
    V2 = np.zeros((nbf, nbf))

    # Cartesian basis one-electron EFP perturbation
    efp_ints = np.zeros((20, nbf, nbf))

    for imp in range(n_mp):
        origin = xyz_mp[imp]

        # get EFP multipole integrals from Psi4
        p4_efp_ints = mints.ao_efp_multipole_potential(origin=origin)
        for pole in range(20):
            efp_ints[pole] = np.asarray(p4_efp_ints[pole])

        # add frag atom Z into multipole charge (when pos'n of atom matches mp)
        for ifr in range(n_fr):
            atoms = efpobj.get_frag_atoms(ifr)
            for iat in range(natoms[ifr]):
                xyz_atom = [atoms[iat]['x'], atoms[iat]['y'], atoms[iat]['z']]
                if np.allclose(xyz_atom, origin, atol=1e-10):
                    val_mp[imp, 0] += atoms[iat]['Z']

        # scale multipole integrals by multipole magnitudes. result goes into V
        for pole in range(20):
            efp_ints[pole] *= -prefacs[pole] * val_mp[imp, pole]
            V2 += efp_ints[pole]

    return V2


def modify_Fock_induced(nbf, efpobj, verbose=1):
    """Returns shared matrix containing the EFP contribution to the potential
    felt by QM atoms, due to EFP induced dipoles, in a SCF procedure.

    """
    # get induced dipoles count, pos'n, values from libefp
    #   dipoles = 3
    n_id = efpobj.get_induced_dipole_count()
    xyz_id = np.asarray(efpobj.get_induced_dipole_coordinates(verbose=verbose)).reshape(n_id, 3)
    val_id = np.asarray(efpobj.get_induced_dipole_values(verbose=verbose)).reshape(n_id, 3)
    val_idt = np.asarray(efpobj.get_induced_dipole_conj_values(verbose=verbose)).reshape(n_id, 3)

    # take average of induced dipole and conjugate
    val_id = (val_id + val_idt) * 0.5

    # EFP induced dipole contribution to the Fock Matrix
    V2 = np.zeros((nbf, nbf))

    # Cartesian basis one-electron EFP perturbation
    field_ints = np.zeros((3, nbf, nbf))

    for iid in range(n_id):
        origin = xyz_id[iid]

        # get electric field integrals from Psi4
        p4_field_ints = mints.electric_field(origin=origin)
        for pole in range(3):
            field_ints[pole] = np.asarray(p4_field_ints[pole])

        # scale field integrals by induced dipole magnitudes. result goes into V
        for pole in range(3):
            field_ints[pole] *= -val_id[iid, pole]
            V2 += field_ints[pole]

    return V2


def field_fn(xyz):
    """Compute electric field from electrons in ab initio part for libefp polarization calculation.

    Parameters
    ----------
    xyz : list
        3 * n_pt (flat) array of points at which to compute electric field

    Returns
    -------
    list
        3 * n_pt (flat) array of electric field at points in `xyz`.

    Notes
    -----
    Function signature defined by libefp, so function uses number of
    basis functions `nbf` and density matrix `efp_density` from global
    namespace.

    """
    global nbf
    global efp_density

    points = np.array(xyz).reshape(-1, 3)
    n_pt = len(points)

    # Cartesian basis one-electron EFP perturbation
    field_ints = np.zeros((3, nbf, nbf))

    # Electric field at points
    field = np.zeros((n_pt, 3))

    for ipt in range(n_pt):
        # get electric field integrals from Psi4
        p4_field_ints = mints.electric_field(origin=points[ipt])

        field[ipt] = [np.vdot(efp_density, np.asarray(p4_field_ints[0])) * 2.0,  # Ex
                      np.vdot(efp_density, np.asarray(p4_field_ints[1])) * 2.0,  # Ey
                      np.vdot(efp_density, np.asarray(p4_field_ints[2])) * 2.0]  # Ez

    field = np.reshape(field, 3 * n_pt)

    return field


ref_V2 = np.array([
 [ -0.02702339455725,    -0.00631509453548,    -0.00000280084677,    -0.00060226624612,     0.00000155158400,  -0.00452046694500,    -0.00000038595163,    -0.00008299120179,     0.00000021380548,    -0.00090142990526,      0.00000473984815,     0.00000183105977,    -0.00091126369988,     0.00000235760871,    -0.00090571433548,    -0.00093899785533,    -0.00186143580968,    -0.00093995668834,    -0.00186166418149],
 [ -0.00631509453548,    -0.02702339455725,    -0.00001910979606,    -0.00410918056805,     0.00001058624630,  -0.02063616591789,    -0.00001267718384,    -0.00272597555850,     0.00000702277450,    -0.01445203384431,      0.00033391900577,     0.00012899688747,    -0.01514481783551,     0.00016609189393,    -0.01475386897318,    -0.00642763371094,    -0.01023119897476,    -0.00663585147245,    -0.01026009701560],
 [ -0.00000280084677,    -0.00001910979606,    -0.02665234730712,     0.00037371007562,     0.00014436865150,  -0.00001859005760,    -0.01317104219809,     0.00038448758263,     0.00014853213109,    -0.00013643690212,     -0.00190980298320,    -0.00014163627448,     0.00010970691227,    -0.00002569550824,    -0.00001652160690,     0.00553694631171,     0.00296253882599,    -0.00592518334643,    -0.00307008036331],
 [ -0.00060226624612,    -0.00410918056805,     0.00037371007562,    -0.02742768609665,     0.00018588404124,  -0.00399742114424,     0.00038448758263,    -0.01396874115447,     0.00019124479202,    -0.00190980298320,      0.00010970691227,    -0.00002569550824,    -0.00553609000895,     0.00005214006927,    -0.00185450039426,    -0.00111418876009,    -0.00223702838075,    -0.00084629056834,    -0.00203426014685],
 [  0.00000155158400,     0.00001058624630,     0.00014436865150,     0.00018588404124,    -0.02699015026797,   0.00001029832686,     0.00014853213109,     0.00019124479202,    -0.01351858713356,    -0.00014163627448,     -0.00002569550824,    -0.00001652160690,     0.00005214006927,    -0.00185450039426,     0.00011345627547,     0.00451469309687,     0.00241810592738,     0.00477138911517,     0.00250189743578],
 [ -0.00452046694500,    -0.02063616591789,    -0.00001859005760,    -0.00399742114424,     0.00001029832686,  -0.02702319749786,    -0.00003768753434,    -0.00796996845458,     0.00002030483034,    -0.01818952603674,      0.00070524995886,     0.00027244649517,    -0.01965271296696,     0.00035079256242,    -0.01882701369086,    -0.00987165499275,    -0.01795740782584,    -0.01115822875565,    -0.01834575971991],
 [ -0.00000038595163,    -0.00001267718384,    -0.01317104219809,     0.00038448758263,     0.00014853213109,  -0.00003768753434,    -0.02503648443824,     0.00199702997314,     0.00077269122299,    -0.00033902442705,     -0.00294678699585,    -0.00039007690599,     0.00030807567722,    -0.00006972255302,    -0.00003443473716,     0.00878465177892,     0.00777812963854,    -0.01154437574140,    -0.00936638912773],
 [ -0.00008299120179,    -0.00272597555850,     0.00038448758263,    -0.01396874115447,     0.00019124479202,  -0.00796996845458,     0.00199702997314,    -0.02918413124002,     0.00099361419288,    -0.00294678699585,      0.00030807567722,    -0.00006972255302,    -0.00831580669109,     0.00013571897621,    -0.00279672819574,    -0.00251085900448,    -0.00821286621429,    -0.00153039204428,    -0.00622386437502],
 [  0.00000021380548,     0.00000702277450,     0.00014853213109,     0.00019124479202,    -0.01351858713356,   0.00002030483034,     0.00077269122299,     0.00099361419288,    -0.02684469027539,    -0.00039007690599,     -0.00006972255302,    -0.00003443473716,     0.00013571897621,    -0.00279672819574,     0.00029057799444,     0.00789015760008,     0.00719868343548,     0.00944135089382,     0.00814913589233],
 [ -0.00090142990526,    -0.01445203384431,    -0.00013643690212,    -0.00190980298320,    -0.00014163627448,  -0.01818952603674,    -0.00033902442705,    -0.00294678699585,    -0.00039007690599,    -0.02563070634460,      0.00066403177471,     0.00035090564283,    -0.00910270453424,     0.00007502850470,    -0.00874245358696,    -0.00913676610260,    -0.01107408168593,    -0.01046477748444,    -0.01146456481073],
 [  0.00000473984815,     0.00033391900577,    -0.00190980298320,     0.00010970691227,    -0.00002569550824,   0.00070524995886,    -0.00294678699585,     0.00030807567722,    -0.00006972255302,     0.00066403177471,     -0.00910270453424,     0.00007502850470,     0.00068756471942,     0.00005040470506,     0.00022274776644,     0.00124025666869,     0.00135413078331,    -0.00068224645268,    -0.00032923928756],
 [  0.00000183105977,     0.00012899688747,    -0.00014163627448,    -0.00002569550824,    -0.00001652160690,   0.00027244649517,    -0.00039007690599,    -0.00006972255302,    -0.00003443473716,     0.00035090564283,      0.00007502850470,    -0.00874245358696,     0.00005040470506,     0.00022274776644,     0.00020687758368,    -0.00412380528180,    -0.00051519173670,     0.00491320446628,     0.00097904308284],
 [ -0.00091126369988,    -0.01514481783551,     0.00010970691227,    -0.00553609000895,     0.00005214006927,  -0.01965271296696,     0.00030807567722,    -0.00831580669109,     0.00013571897621,    -0.00910270453424,      0.00068756471942,     0.00005040470506,    -0.02840235120105,     0.00033061285791,    -0.00923711128151,    -0.00458459601546,    -0.01138951947581,    -0.00454371602298,    -0.01120759511573],
 [  0.00000235760871,     0.00016609189393,    -0.00002569550824,     0.00005214006927,    -0.00185450039426,   0.00035079256242,    -0.00006972255302,     0.00013571897621,    -0.00279672819574,     0.00007502850470,      0.00005040470506,     0.00022274776644,     0.00033061285791,    -0.00923711128151,     0.00037744020930,     0.00095088751145,     0.00091755913622,     0.00066686324895,     0.00079498664458],
 [ -0.00090571433548,    -0.01475386897318,    -0.00001652160690,    -0.00185450039426,     0.00011345627547,  -0.01882701369086,    -0.00003443473716,    -0.00279672819574,     0.00029057799444,    -0.00874245358696,      0.00022274776644,     0.00020687758368,    -0.00923711128151,     0.00037744020930,    -0.02691937643507,    -0.00793049330280,    -0.01147295613562,    -0.00845374029895,    -0.01156033431781],
 [ -0.00093899785533,    -0.00642763371094,     0.00553694631171,    -0.00111418876009,     0.00451469309687,  -0.00987165499275,     0.00878465177892,    -0.00251085900448,     0.00789015760008,    -0.00913676610260,      0.00124025666869,    -0.00412380528180,    -0.00458459601546,     0.00095088751145,    -0.00793049330280,    -0.01785633292778,    -0.01175654591020,    -0.00144863365096,    -0.00543904115350],
 [ -0.00186143580968,    -0.01023119897476,     0.00296253882599,    -0.00223702838075,     0.00241810592738,  -0.01795740782584,     0.00777812963854,    -0.00821286621429,     0.00719868343548,    -0.01107408168593,      0.00135413078331,    -0.00051519173670,    -0.01138951947581,     0.00091755913622,    -0.01147295613562,    -0.01175654591020,    -0.01842598268335,    -0.00600898660138,    -0.01416862694275],
 [ -0.00093995668834,    -0.00663585147245,    -0.00592518334643,    -0.00084629056834,     0.00477138911517,  -0.01115822875565,    -0.01154437574140,    -0.00153039204428,     0.00944135089382,    -0.01046477748444,     -0.00068224645268,     0.00491320446628,    -0.00454371602298,     0.00066686324895,    -0.00845374029895,    -0.00144863365096,    -0.00600898660138,    -0.02521907195360,    -0.01660151455045],
 [ -0.00186166418149,    -0.01026009701560,    -0.00307008036331,    -0.00203426014685,     0.00250189743578,  -0.01834575971991,    -0.00936638912773,    -0.00622386437502,     0.00814913589233,    -0.01146456481073,     -0.00032923928756,     0.00097904308284,    -0.01120759511573,     0.00079498664458,    -0.01156033431781,    -0.00543904115350,    -0.01416862694275,    -0.01660151455045,    -0.02521511777687]])



mol = psi4.geometry("""
units bohr
0 1
O1     0.000000000000     0.000000000000     0.224348285559
H2    -1.423528800232     0.000000000000    -0.897393142237
H3     1.423528800232     0.000000000000    -0.897393142237
symmetry c1
no_com
no_reorient
""")

# <-- efp
# [Kaliman:2013:2284] Fig. 4 -- Initialize EFP
efpmol = pylibefp.core.efp()
# [Kaliman:2013:2284] Fig. 4 -- Set fragment coordinates
frags = ['h2o', 'nh3', 'nh3']
efpmol.add_potential(frags)
efpmol.add_fragment(frags)
efpmol.set_frag_coordinates(0, 'xyzabc', [-4.014110144291,  2.316749370493, -1.801514729931, -2.902133, 1.734999, -1.953647])
efpmol.set_frag_coordinates(1, 'xyzabc', [ 1.972094713645,  3.599497221584,  5.447701074734, -1.105309, 2.033306, -1.488582])
efpmol.set_frag_coordinates(2, 'xyzabc', [-7.876296399270, -1.854372164887, -2.414804197762,  2.526442, 1.658262, -2.742084])
efpmol.prepare()
efpmol.set_opts({}, append='psi')
efpmol.set_electron_density_field_fn(field_fn)
# --> efp

psi4.set_options({'basis': '6-31g*',
                  'scf_type': 'pk',
                  'e_convergence': 1e-8})

# Set defaults
maxiter = 40
E_conv = 1.0E-6
D_conv = 1.0E-3

# Integral generation from Psi4's MintsHelper
wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option('BASIS'))
t = time.time()
mints = psi4.core.MintsHelper(wfn.basisset())
S = np.asarray(mints.ao_overlap())

# Get nbf and ndocc for closed shell molecules
nbf = S.shape[0]
ndocc = wfn.nalpha()

print('\nNumber of occupied orbitals: %d' % ndocc)
print('Number of basis functions: %d' % nbf)

# Run a quick check to make sure everything will fit into memory
I_Size = (nbf ** 4) * 8.e-9
print("\nSize of the ERI tensor will be %4.2f GB." % I_Size)

# Estimate memory usage
memory_footprint = I_Size * 1.5
if I_Size > numpy_memory:
    psi4.core.clean()
    raise Exception("Estimated memory utilization (%4.2f GB) exceeds numpy_memory \
                    limit of %4.2f GB." % (memory_footprint, numpy_memory))

# Compute required quantities for SCF
V = np.asarray(mints.ao_potential())
T = np.asarray(mints.ao_kinetic())
I = np.asarray(mints.ao_eri())

print('\nTotal time taken for integrals: %.3f seconds.' % (time.time() - t))

t = time.time()

# Build H_core
H = T + V

# <-- efp: add in permanent moment contribution and cache
Vefp = modify_Fock_permanent(mol, nbf, efpmol)
assert(psi4.compare_integers(1, np.allclose(Vefp, ref_V2), 'EFP permanent Fock contrib'))
H = H + Vefp
Horig = H.copy()
set_qm_atoms(mol, efpmol)
# --> efp

# Orthogonalizer A = S^(-1/2) using Psi4's matrix power.
A = mints.ao_overlap()
A.power(-0.5, 1.e-16)
A = np.asarray(A)

# Calculate initial core guess
Hp = A.dot(H).dot(A)
e, C2 = np.linalg.eigh(Hp)
C = A.dot(C2)
Cocc = C[:, :ndocc]
D = np.einsum('pi,qi->pq', Cocc, Cocc)

print('\nTotal time taken for setup: %.3f seconds' % (time.time() - t))

print('QM/EFP: iterating Total Energy including QM/EFP Induction')
t = time.time()
E = 0.0
Enuc = mol.nuclear_repulsion_energy()
Eold = 0.0
Dold = np.zeros_like(D)

for SCF_ITER in range(1, maxiter + 1):

    # <-- efp: add contribution to Fock matrix
    verbose_dipoles = 1 if (SCF_ITER == 1) else 0
    # [Kaliman:2013:2284] Fig. 4 -- Compute electric field from wavefunction
    # [Kaliman:2013:2284] Fig. 4 -- Compute electric field from induced dipoles
    Vefp = modify_Fock_induced(nbf, efpmol, verbose=verbose_dipoles)
    H = Horig.copy() + Vefp
    # --> efp

    # Build fock matrix
    J = np.einsum('pqrs,rs->pq', I, D)
    K = np.einsum('prqs,rs->pq', I, D)
    F = H + J * 2 - K

    diis_e = np.einsum('ij,jk,kl->il', F, D, S) - np.einsum('ij,jk,kl->il', S, D, F)
    diis_e = A.dot(diis_e).dot(A)

    # SCF energy and update
    # [Kaliman:2013:2284] Fig. 4 -- Compute QM wavefunction
    SCF_E = np.einsum('pq,pq->', F + H, D) + Enuc
    dRMS = np.mean(diis_e**2)**0.5

    # <-- efp: add contribution to energy
    efp_density = D
    # [Kaliman:2013:2284] Fig. 4 -- Compute EFP induced dipoles
    efp_wfn_dependent_energy = efpmol.get_wavefunction_dependent_energy()
    SCF_E += efp_wfn_dependent_energy
    # --> efp

    print('SCF Iteration %3d: Energy = %4.16f   dE = % 1.5E   dRMS = %1.5E   dEFP = %12.8f'
          % (SCF_ITER, SCF_E, (SCF_E - Eold), dRMS, efp_wfn_dependent_energy))
    if (abs(SCF_E - Eold) < E_conv) and (dRMS < D_conv):
        break

    Eold = SCF_E
    Dold = D

    # Diagonalize Fock matrix
    Fp = A.dot(F).dot(A)
    e, C2 = np.linalg.eigh(Fp)
    C = A.dot(C2)
    Cocc = C[:, :ndocc]
    D = np.einsum('pi,qi->pq', Cocc, Cocc)

    if SCF_ITER == maxiter:
        clean()
        raise Exception("Maximum number of SCF cycles exceeded.")


# <-- efp
efpmol.compute()
efpene = efpmol.get_energy(label='psi')
# [Kaliman:2013:2284] Fig. 4 -- Compute one electron EFP contributions to Hamiltonian
efp_wfn_independent_energy = efpene['total'] - efpene['ind']
SCF_E += efp_wfn_independent_energy
print(efpmol.energy_summary(scfefp=SCF_E, label='psi'))
# --> efp

print('Total time for SCF iterations: %.3f seconds \n' % (time.time() - t))

# references confirmed against Q-Chem & Psi4
assert(psi4.compare_values( 0.2622598847, efpene['total'] - efpene['ind'], 6, 'EFP corr to SCF'))
assert(psi4.compare_values(-0.0117694790, efpene['ind'], 6, 'QM-EFP Indc'))
assert(psi4.compare_values(-0.0021985285, efpene['disp'], 6, 'EFP-EFP Disp'))
assert(psi4.compare_values( 0.0056859871, efpene['exch'], 6, 'EFP-EFP Exch'))
assert(psi4.compare_values( 0.2504904057, efpene['total'], 6, 'EFP-EFP Totl'))
assert(psi4.compare_values(-76.0139362744, SCF_E, 6, 'SCF'))
efpmol.clean()
