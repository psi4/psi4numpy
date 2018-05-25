"""
Helper classes and functions for the RESP program. 

Assists in generating van der Waals surface, computing the electrostatic
potential with Psi4, and adding constraints for two-stage fitting procedure.
"""

__authors__   =  "Asim Alenaizan"
__credits__   =  ["Asim Alenaizan"]

__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__   = "BSD-3-Clause"
__date__      = "2018-04-28"

import numpy as np

class helper_VDW_surface(object):
    """
    A script to generate van der Waals surface of molecules.
    """

    #Van der Waals radii (in angstrom) are taken from GAMESS.
    vdw_r = {'H': 1.20, 'HE': 1.20,
             'LI': 1.37, 'BE': 1.45, 'B': 1.45, 'C': 1.50,
             'N': 1.50, 'O': 1.40, 'F': 1.35, 'NE': 1.30,
             'NA': 1.57, 'MG': 1.36, 'AL': 1.24, 'SI': 1.17,
             'P': 1.80, 'S': 1.75, 'CL': 1.70}

    def _surface(self, n):
        """Computes approximately n points on unit sphere. Code adapted from GAMESS.

        Parameters
        ----------
        n : int
            approximate number of requested surface points

        Returns
        -------
        np.array
            nupmy array of xyz coordinates of surface points
        """
            
        u = []
        eps = 1e-10
        nequat = int(np.sqrt(np.pi*n))
        nvert = int(nequat/2)
        nu = 0
        for i in range(nvert+1):
            fi = np.pi*i/nvert
            z = np.cos(fi)
            xy = np.sin(fi)
            nhor = int(nequat*xy+eps)
            if nhor < 1:
                nhor = 1
            for j in range(nhor):
                fj = 2*np.pi*j/nhor
                x = np.cos(fj)*xy
                y = np.sin(fj)*xy
                if nu >= n:
                    return np.array(u)
                nu += 1
                u.append([x, y, z])
        return np.array(u) 

    def vdw_surface(self, coordinates, elements, scale_factor, density, input_radii):
        """Computes points outside the van der Walls' surface of molecules.

        Parameters
        ----------
        coordinates : np.array
            cartesian coordinates of the nuclei, in units of angstrom
        elements : list
            The symbols (e.g. C, H) for the atoms
        scale_factor : float
            The points on the molecular surface are set at a distance of
            scale_factor * vdw_radius away from each of the atoms.
        density : float
            The (approximate) number of points to generate per square angstrom
            of surface area. 1.0 is the default recommended by Kollman & Singh.
        input_radii : dict
            dictionary of user's defined VDW radii

        Returns
        -------
        radii : dict
            A dictionary of scaled VDW radii
        surface_points : np.array
            array of the coordinates of the points on the surface
        """
        radii = {}
        surface_points = []
        # scale radii
        for i in elements:
            if i in radii.keys():
                continue
            if i in input_radii.keys():
                radii[i] = input_radii[i] * scale_factor
            elif i in self.vdw_r.keys():
                radii[i] = self.vdw_r[i] * scale_factor
            else:
                raise KeyError('%s is not a supported element; ' %i
                             + 'use the "RADIUS" option to add '
                             + 'its van der Waals radius.')
        for i in range(len(coordinates)):
            # calculate points
            n_points = int(density * 4.0 * np.pi* np.power(radii[elements[i]], 2))
            dots = self._surface(n_points)
            dots = coordinates[i] + radii[elements[i]] * dots
            for j in range(len(dots)):
                save = True
                for k in range(len(coordinates)):
                    if i == k:
                        continue
                    # exclude points within the scaled VDW radius of other atoms
                    d = np.linalg.norm(dots[j] - coordinates[k])
                    if d < radii[elements[k]]:
                        save = False
                        break
                if save:
                    surface_points.append(dots[j])
        self.radii = radii
        self.shell = np.array(surface_points)

class helper_stage2(object):
    """
    A helper script to facilitate the use of constraints for two-stage fitting.
    """

    def _get_stage2_atoms(self, molecule, cutoff=1.2):
        """Determines atoms for second stage fit. The atoms
           are identified as C-H bonded groups based and a cutoff distance.

        Parameters
        ----------
        molecule : psi4.Molecule instance 

        cutoff : float, optional
            a cutoff distance in Angstroms, exclusive

        Returns
        -------
        groups : dict
            a dictionary whose keys are the indecies+1 of carbon
            atoms and whose elements are the indecies+1 of the
            connected hydrogen atoms.
        """
        bohr_to_angstrom = 0.52917721092
        coordinates = molecule.geometry()
        coordinates = coordinates.np.astype('float')*bohr_to_angstrom
        symbols = []
        for i in range(molecule.natom()):
            symbols.append(molecule.symbol(i))
        l = np.zeros(molecule.natom())
        groups = {}
        for i in range(molecule.natom()-1):
            hydrogens = []
            for j in range(i+1, molecule.natom()):
                if (symbols[i] == 'C' and symbols[j] == 'H') or \
                   (symbols[j] == 'C' and symbols[i] == 'H'):
                    d = np.linalg.norm(coordinates[i]-coordinates[j])
                    if d < cutoff:
                        if symbols[i] == 'C': 
                            if i+1 not in groups.keys():
                                groups[i+1] = []
                            groups[i+1].append(j+1)
                        if symbols[j] == 'C':
                            if j+1 not in groups.keys():
                                groups[j+1] = []
                            groups[j+1].append(i+1)

        return groups


    def set_stage2_constraint(self, molecule, charges, options, cutoff=1.2):
        """Sets default constraints for the second stage fit.

        The default constraints are the following:
        Atoms that are excluded from the second stage fit are constrained
        to their charges from the first stage fit. C-H groups that have
        bonds shorter than the cutoff distance are refitted and the
        hydrogen atoms connected to the same carbon are constrained to
        have identical charges. This calls self._get_stage2_atoms.

        Parameters
        ----------
        molecule : psi4.Molecule instance

        charges : np.array
            array containing the charges from the first stage fit
        options : dict
            dictionary of the fitting options. To be modified in place.
        cutoff : float, optional
            cutoff distance in Angstroms, exclusive

        Return
        ------
        None
        """
        second_stage = self._get_stage2_atoms(molecule, cutoff=cutoff)
        atoms = list(range(1, molecule.natom()+1))
        constraint_group = []
        for i in second_stage.keys():
            atoms.remove(i)
            group = []
            for j in second_stage[i]:
                atoms.remove(j)
                group.append(j)
            constraint_group.append(group)
        constraint_charge = []
        for i in atoms:
            constraint_charge.append([charges[i-1], [i]])
        options['constraint_charge'] = constraint_charge
        options['constraint_group'] = constraint_group


    def stage2_intermolecular_constraint(self, molecules, cutoff=1.2):
        """Determines the default intermolecular constraint for multi-molecular 
        fit, in the second stage fit.

        The default is that the equivalent carbon atoms in the different
        molecules are made equivalent, and only one of the hydrogens
        in a group is made equivalent with the corresponding hydrogen
        in the other molecule. This calls self.get_stage2_atoms and use
        the given cutoff distance.

        Parameters
        ----------
        molecules : list of psi4.Molecule
            list of psi4.Molecule instances.
        cutoff : float, optional
            cutoff distance in Angstroms, exclusive

        Return
        ------
        intermolecular_constraint : dict
            a dictionary of intermolecular constraint   
        """
        inter_constraint = []
        for mol in range(len(molecules)):
            equals = [mol,[]]
            second_stage = self._get_stage2_atoms(molecules[mol], cutoff=cutoff)
            for i in second_stage.keys():
                equals[1].append(i)
                try:
                    equals[1].append(second_stage[i][0])
                except:
                    pass
            inter_constraint.append(equals)
        inter = []
        for i in range(1, len(inter_constraint)):
            inter.append([inter_constraint[0], inter_constraint[i]])
        intermolecular_constraint = {'EQUAL': inter} 
        self.intermolecular_constraint = intermolecular_constraint
