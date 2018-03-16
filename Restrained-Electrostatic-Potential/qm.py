def psi4_esp(method, basis, molecule):
    """ Computation of QM electrostatic potential with psi4
        input:
               method: string of QM method
               basis: string of basis set
               molecule: an instance of psi4 Molecule class
        output:
               wfn.oeprop.Vvals(): numpy array of QM electrostatic potential
        Note: grid is available in grid.dat
     """
    import psi4
    mol = psi4.geometry(molecule.create_psi4_string_from_molecule())
    psi4.set_options({'basis': basis})
    e, wfn = psi4.prop(method, properties=['GRID_ESP'], return_wfn=True)
    psi4.core.clean()
    return wfn.oeprop.Vvals()
