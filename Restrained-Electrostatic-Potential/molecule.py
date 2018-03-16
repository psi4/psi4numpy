import numpy as np
bohr_to_angstrom = 0.52917721092

def get_stage2_atoms(molecule, cutoff=1.2):
    """Function to determine atoms for second stage fit. The atoms
       are identified as C-H bonded groups based and a cutoff distance.
    input:
          molecule: an instance of psi4 Molecule class
          cutoff: a cutoff distance in Angstroms, exclusive
    output:
          groups: a dictionary whose keys are the indecies+1 of carbon
                  atoms and whose elements are the indecies+1 of the
                  connected hydrogen atoms.
    """
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


def set_stage2_constraint(molecule, charges, options, cutoff=1.2):
    """Function to set default constraints for the second stage fit.
       The default constraints are the following:
       Atoms that are excluded from the second stage fit are constrained
       to their charges from the first stage fit. C-H groups that have
       bonds shorter than the cutoff distance are refitted and the
       hydrogen atoms connected to the same carbon are constrained to
       have identical charges. This calls get_stage2_atoms.
    input:
        molecule: an instance of Psi4 Molecule class
        options: Dictionary for the fitting options. To be modified in place.
        cutoff: a cutoff distance in Angstroms, exclusive
    output:
        None. options modified in place
    """
    second_stage = get_stage2_atoms(molecule, cutoff=cutoff)
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


def stage2_intermolecular_constraint(molecules, cutoff=1.2):
    """Function to determine the default intermolecular constraint
       for multi-molecular fit, in the second stage fit.
       The default is that the equivalent carbon atoms in the different
       molecules are made equivalent, and only one of the hydrogens
       in a group is made equivalent with the corresponding hydrogen
       in the other molecule. This calls self.get_stage2_atoms and use
       the given cutoff distance
    input:
       molecules: list of molecule instances.
       cutoff: a cutoff distance in Angstroms, exclusive
    output:
       intermolecular_constraint: A dictionary of intermolecular constraint   
    """
    inter_constraint = []
    for mol in range(len(molecules)):
        equals = [mol,[]]
        second_stage = get_stage2_atoms(molecules[mol], cutoff=cutoff)
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
    return intermolecular_constraint
