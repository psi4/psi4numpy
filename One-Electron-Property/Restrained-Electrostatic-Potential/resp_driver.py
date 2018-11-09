"""
Driver for the RESP code.
"""

__authors__   =  "Asim Alenaizan"
__credits__   =  ["Asim Alenaizan"]

__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__   = "BSD-3-Clause"
__date__      = "2018-04-28"

import numpy as np
import os
from espfit import *
from resp_helper import *

bohr_to_angstrom = 0.52917721092

def resp(molecules, options_list=[], intermol_constraints={}):
    """RESP code driver.

    Parameters
    ---------- 
    molecules : list
        list of psi4.Molecule instances
    options_list : list, optional
        list of dictionaries of user's defined options
    intermol_constraints : dict, optional
        dictionary of options for multi-molecules fitting

    Returns
    -------
    charges : list
        list of charges

    Note
    ----
    output files : mol_results.dat: fitting results
                   mol_grid.dat: grid points in molecule.units
                   mol_grid_esp.dat: QM esp valuese in a.u. 
    """
    # Check options
    # Large case keys: resp options
    # Small case key: internal data
    check = {}
    for i in intermol_constraints.keys():
        check[i.upper()] = intermol_constraints[i]
    intermol_constraints = check
    if not ('CHARGE' in intermol_constraints.keys()):
        intermol_constraints['CHARGE'] = [] 
    if not ('EQUAL' in intermol_constraints.keys()):
        intermol_constraints['EQUAL'] = []

    # Check options for first molecule
    check_options = {}
    for i in sorted(options_list[0].keys()):
        check_options[i.upper()] = options_list[0][i]
    options = check_options
    # VDW surface options
    if not ('ESP' in options.keys()):
        options['ESP'] = []
    if not ('GRID' in options.keys()):
        options['GRID'] = []
    if not ('N_VDW_LAYERS' in options.keys()):
        options['N_VDW_LAYERS'] = 4
    if not ('VDW_SCALE_FACTOR' in options.keys()):
        options['VDW_SCALE_FACTOR'] = 1.4
    if not ('VDW_INCREMENT' in options.keys()):
        options['VDW_INCREMENT'] = 0.2
    if not ('VDW_POINT_DENSITY' in options.keys()):
        options['VDW_POINT_DENSITY'] = 1.0
    # Hyperbolic restraint options
    if not ('WEIGHT' in options.keys()):
        options['WEIGHT'] = 1
    if not ('RESTRAINT' in options.keys()):
        options['RESTRAINT'] = True
    if options['RESTRAINT']:
        if not ('RESP_A' in options.keys()):
            options['RESP_A'] = 0.0005
        if not ('RESP_B' in options.keys()):
            options['RESP_B'] = 0.1
        if not ('IHFREE' in options.keys()):
            options['IHFREE'] = True
        if not ('TOLER' in options.keys()):
            options['TOLER'] = 1e-5
        if not ('MAX_IT' in options.keys()):
            options['MAX_IT'] = 25

    # QM options
    if not ('METHOD_ESP' in options.keys()):
        options['METHOD_ESP'] = 'scf'
    if not ('BASIS_ESP' in options.keys()):
        options['BASIS_ESP'] = '6-31g*'

    options_list[0] = options

    final_options_list = []
    n_atoms = []
    symbols_list = []
    for mol in range(len(molecules)):
        check_options = {}
        for i in options_list[mol].keys():
            check_options[i.upper()] = options_list[mol][i]
        options = check_options
        # VDW surface options
        if not ('RADIUS' in options.keys()):
            options['RADIUS'] = {}
        radii = {}
        for i in options['RADIUS'].keys():
            radii[i.upper()] = options['RADIUS'][i]
        options['RADIUS'] = radii

        # Constraint options
        if not ('CONSTRAINT_CHARGE' in options.keys()):
            options['CONSTRAINT_CHARGE'] = []
        if not ('CONSTRAINT_GROUP' in options.keys()):
            options['CONSTRAINT_GROUP'] = []
        if not ('CONSTRAINT_EQUAL' in options.keys()):
            options['CONSTRAINT_EQUAL'] = []
    
        if mol > 0:
            for i in final_options_list[0].keys():
                if i not in options.keys() and i.isupper():
                    options[i] = final_options_list[0][i] 

        options['mol_charge'] = molecules[mol].molecular_charge()
        n_atoms.append(molecules[mol].natom())
        coordinates = molecules[mol].geometry()
        coordinates = coordinates.np.astype('float')*bohr_to_angstrom
        options['coordinates'] = coordinates
        symbols = []
        for i in range(n_atoms[-1]):
            symbols.append(molecules[mol].symbol(i))
        options['symbols'] = symbols
        symbols_list.append(symbols)

        if options['GRID']:
            # Read grid points
            points = np.loadtxt(options['GRID'])
            np.savetxt('grid.dat', points, fmt='%15.10f')
            if 'Bohr' in str(molecules[mol].units):
                points *= bohr_to_angstrom

        else:
            # Get the points at which we're going to calculate the ESP surface
            points = []
            surface = helper_VDW_surface()
            for i in range(options['N_VDW_LAYERS']):
                scale_factor = options['VDW_SCALE_FACTOR'] + i * options['VDW_INCREMENT']
                surface.vdw_surface(coordinates, symbols, scale_factor,
                                    options['VDW_POINT_DENSITY'], options['RADIUS'])
                points.append(surface.shell)
            radii = surface.radii
            points = np.concatenate(points)
            if 'Bohr' in str(molecules[mol].units):
                points /= bohr_to_angstrom
                np.savetxt('grid.dat', points, fmt='%15.10f')
                points *= bohr_to_angstrom
            else:
                np.savetxt('grid.dat', points, fmt='%15.10f')

        # Calculate ESP values at the grid
        if options['ESP']:
            # Read electrostatic potential values
            options['esp_values'] = np.loadtxt(options['ESP'])
            np.savetxt('grid_esp.dat', options['esp_values'], fmt='%15.10f')
        else:
            import psi4
            psi4.core.set_active_molecule(molecules[mol])
            psi4.set_options({'basis': options['BASIS_ESP']})
            psi4.prop(options['METHOD_ESP'], properties=['GRID_ESP'])
            options['esp_values'] = np.loadtxt('grid_esp.dat')
            psi4.core.clean()
            
        os.system("mv grid.dat %i_%s_grid.dat" %(mol+1, molecules[mol].name()))
        os.system("mv grid_esp.dat %i_%s_grid_esp.dat" %(mol+1, molecules[mol].name()))
        # Build a matrix of the inverse distance from each ESP point to each nucleus
        invr = np.zeros((len(points), len(coordinates)))
        for i in range(invr.shape[0]):
            for j in range(invr.shape[1]):
                invr[i, j] = 1/np.linalg.norm(points[i]-coordinates[j])
        options['invr'] = invr*bohr_to_angstrom # convert to atomic units
        options['coordinates'] /= bohr_to_angstrom # convert to angstroms

        final_options_list.append(options)
    # Calculate charges
    qf, labelf, notes = fit(final_options_list, intermol_constraints)
    index = 0
    charges = []
    
    # Exstract the charges
    for mol in range(len(molecules)):
        q = []
        for i in qf:
            q.append(i[index:index+n_atoms[mol]])
        index += n_atoms[mol]
        charges.append(q)

    for mol in range(len(molecules)):
        options = final_options_list[mol]
        # Write the resules to disk
        f = open(str(mol+1) + '_' + molecules[mol].name() + "_results.out", "w")
        f.write("\n Electrostatic potential parameters\n")
        f.write("\n Geometry (see% i_%s.xyz in Angstrom)\n" %(mol+1, molecules[mol].name()))
        f.write("\n Grid information (see %i_%s_grid.dat in %s)\n" %(mol+1, molecules[mol].name(), molecules[mol].units))
        f.write("     van der Waals radii (Angstrom):\n")
        for i, j in radii.items():
            f.write("                                %8s%8.3f\n" %(i, j/scale_factor))
        f.write("     Number of VDW layers:             %d\n" %(options["N_VDW_LAYERS"]))
        f.write("     VDW scale facotr:                 %.3f\n" %(options["VDW_SCALE_FACTOR"]))
        f.write("     VDW increment:                    %.3f\n" %(options["VDW_INCREMENT"]))
        f.write("     VDW point density:                %.3f\n" %(options["VDW_POINT_DENSITY"]))
        f.write("     Number of grid points:            %d\n" %len(options['esp_values']))

        f.write("\n Quantum electrostatic potential (see %i_%s_grid_esp.dat)\n" %(mol+1,molecules[0].name()))
        f.write("     ESP method:                       %s\n" %options['METHOD_ESP'])
        f.write("     ESP basis set:                    %s\n" %options['BASIS_ESP'])

        f.write("\n Constraints\n")
        if options['CONSTRAINT_CHARGE']:
            f.write("     Charge constraints\n")
            for i in options['CONSTRAINT_CHARGE']:
                f.write("         Total charge of %8.5f on the set" %i[0])
                for j in i[1]:
                    f.write("%4d" %j)
                f.write("\n")
        if options['CONSTRAINT_GROUP'] or options['CONSTRAINT_EQUAL']:
            f.write("     Equality constraints\n")
            f.write("         Equal charges on atoms\n")
            for i in options['CONSTRAINT_GROUP']:
                f.write("                              ")
                for j in i:
                    f.write("%4d" %j)
                f.write("\n")
            for i in options['CONSTRAINT_EQUAL']:
                for j in range(len(i)):
                    f.write("                              ")
                    f.write("%4d%4d" %(i[0][j], i[1][j]))
                    f.write("\n")
        if intermol_constraints['CHARGE'] or intermol_constraints['EQUAL']:
            f.write('\n     Intermolecular constraints\n')
            if intermol_constraints['CHARGE']:
                f.write('         Charge constraints\n')
                for i in intermol_constraints['CHARGE']:
                    f.write('             Total charge of %8.5f on the set:' %i[0])
                    for j in i[1]:
                        f.write('\n                 molecule %4d, atoms' %j[0])
                        for k in j[1]:
                            f.write('%4d' %k)
                    f.write('\n')
            if intermol_constraints['EQUAL']:
                f.write('         Equality constraints\n')
                f.write('             Equal charges on\n')
                for i in intermol_constraints['EQUAL']:
                    f.write('                 ')
                    f.write('molecule %4d, atoms' %i[0][0])
                    for j in i[0][1]:
                        f.write('%4d' %j)
                    f.write('\n                 molecule %4d, atoms' %i[1][0])
                    for j in i[1][1]:
                        f.write('%4d' %j)
                    f.write('\n\n')
        f.write("\n Restraint\n")
        if options['RESTRAINT']:
            f.write("     Hyperbolic restraint to a charge of zero\n")
            if options['IHFREE']:
                f.write("     Hydrogen atoms are not restrained\n")
            f.write("     resp_a:                           %.4f\n" %(options["RESP_A"]))
            f.write("     resp_b:                           %.4f\n" %(options["RESP_B"]))
        f.write("\n Fit\n")
        for i in notes:
            if i:
                f.write(i+'\n')
        f.write("\n Electrostatic Potential Charges\n")
        f.write("   Center  Symbol")
        for i in labelf:
            f.write("%10s" %i)
        f.write("\n")
        for i in range(n_atoms[mol]):
            f.write("   %5d    %s     " %(i+1, symbols_list[mol][i]))
            for j in charges[mol]:
                f.write("%10.5f" %j[i])
            f.write("\n")
        f.write(" Total Charge:    ")
        for i in charges[mol]:
            f.write("%10.5f" %np.sum(i))
        f.write('\n')
        f.close()

    return charges
