def psi4_esp(method, basis, molecule):
    """ Computes QM electrostatic potential

    Parameters
    ----------
    method : str 
        QM method
    basis : str
        basis set
    molecule : psi4.Molecule instance

    Returns
    -------
    np.array
        QM electrostatic potential.

    Note
    -----
    Psi4 read grid information from grid.dat file
    """
    import psi4
    import numpy as np
    mol = psi4.geometry(molecule.create_psi4_string_from_molecule())
    psi4.set_options({'basis': basis})
    e, wfn = psi4.prop(method, properties=['GRID_ESP'], return_wfn=True)
    psi4.core.clean()
    return np.loadtxt('grid_esp.dat')
