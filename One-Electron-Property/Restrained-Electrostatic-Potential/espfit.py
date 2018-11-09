"""
Fitting procedure for RESP charges.

Reference: 
Equations taken from [Bayly:93:10269].
"""

__authors__   =  "Asim Alenaizan"
__credits__   =  ["Asim Alenaizan"]

__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__   = "BSD-3-Clause"
__date__      = "2018-04-28"

import numpy as np
import copy

def esp_solve(a, b):
    """Solves for point charges: A*q = B

    Parameters
    ----------
    a : np.array
        array of matrix A
    b : np.array
        array of matrix B
    
    Return
    ------
    q : np.array
        array of charges
    """
    q = np.linalg.solve(a, b)
    # Warning for near singular matrix
    # in case np.linalg.solve does not detect singularity
    note = ''
    if np.linalg.cond(a) > 1/np.finfo(a.dtype).eps:
        note = "Possible fit problem; singular matrix"
    return q, note

def restraint(q, akeep, resp_a, resp_b, ihfree, symbols, n_atoms, n_constraints):
    """Adds hyperbolic restraint to matrix A

    Parameters
    ---------- 
    q : np.array
        array of charges
    akeep : np.array
        array of unrestrained A matrix
    resp_a : list
        list of floats of restraint scale a for each molecule
    resp_b : list
        list of floats of restraint parabola tightness b for each molecule
    ihfree : list
        list of bools on whether hydrogen excluded or included in restraint for each molecule
    symbols : list
        list of arrays of element symbols for each molecule
    n_atoms : list
        list of the number of atoms in each molecule
    n_constraints : list
        list of the number of constraints for each molecule

    Returns
    -------
    a : np.array
        restrained A array
    """

    # hyperbolic Restraint
    # [Bayly:93:10271] (Eqs. 10, 13)
    a = copy.deepcopy(akeep)
    n_mol = len(n_atoms)
    index = 0
    index_q = 0
    for mol in range(n_mol):
        for i in range(n_atoms[mol]):
            if not ihfree[mol] or symbols[mol][i] != 'H':
                a[index+i, index+i] = akeep[index+i, index+i]\
                + resp_a[mol]/np.sqrt(q[index_q]**2 + resp_b[mol]**2)
            index_q += 1
        index += n_atoms[mol] + n_constraints[mol]
    return a

def iterate(q, akeep, b, resp_a, resp_b, ihfree, symbols, toler,\
            maxit, n_atoms, n_constraints, indices):
    """Iterates the RESP fitting procedure

    Parameters
    ----------
    q : np.array
        array of initial charges 
    akeep : np.array
        array of unrestrained A matrix
    b : np.array
        array of matrix B
    resp_a : list
        list of floats of restraint scale a for each molecule
    resp_b : list
        list of floats of restraint parabola tightness b for each molecule
    ihfree : list
        list of bools on whether hydrogen excluded or included in restraint for each molecule
    symbols : list
        list of arrays of element symbols for each molecule
    toler : float
        tolerance for charges in the fitting
    maxit : int
        maximum number of iterations
    n_atoms : list
        list of the number of atoms in each molecule
    n_constraints : list
        list of the number of constraints for each molecule
    indices : np.array
        array of the indices for the atoms in the A and B matrices

    Returns
    -------
    q : np.array
        array of the fitted charges
    """
    n_mols = len(n_atoms)
    qkeep = q[indices]
    niter = 0
    difm = 1
    while difm > toler and niter < maxit:
        index = 0
        niter += 1
        a = restraint(q[indices], akeep, resp_a, resp_b, ihfree,\
                      symbols, n_atoms, n_constraints)
        q, note = esp_solve(a, b)
        q_q = q[indices]
        difm = 0
            
        for i in range(len(q_q)):
            dif = (q_q[i]-qkeep[i])**2
            if difm < dif:
                difm = dif
        qkeep = copy.deepcopy(q_q)
        difm = np.sqrt(difm)
    if difm > toler:
        note += '\nCharge fitting did not converge; ' +\
               'try increasing the maximum number of iterations to ' +\
               '> %i.' %maxit
    return q_q, note

def intramolecular_constraints(constraint_charge, constraint_equal, constraint_groups):
    """Extracts intramolecular constraints from user constraint input

    Parameters
    ----------
    constraint_charge : list
        list of lists of charges and atom indices list
        e.g. [[0, [1, 2]], [1, [3, 4]]]
        The sum of charges on 1 and 2 will equal 0
        The sum of charges on 3 and 4 will equal 1
    constraint_equal : list
        list of lists of two lists of indices of atoms to 
        have equal charge element by element
        e.g. [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        atoms 1 and 3 will have equal charge
        atoms 2 and 4 will have equal charge
        and similarly for 5, 6, 7 and 8
    constraint_group : list
        list of lists of indices of atoms to have equal charge
        e.g. [[1, 2], [3, 4]]
        atoms 1 and 2 will have equal charge
        atoms 3 and 4 will have equal charge

    Returns
    -------
    constrained_charges : list
        list of fixed charges 
    constrained_indices : list
        list of lists of indices of atoms in a constraint
        negative number before an index means
        the charge of that atom will be subtracted.
    
    Notes
    -----
    Atom indices starts with 1 not 0.
    Total charge constraint is added by default for the first molecule.
    """                
    constrained_charges = []
    constrained_indices = []
    for i in constraint_charge:
        constrained_charges.append(i[0])
        group = []
        for k in i[1]:
            group.append(k)
        constrained_indices.append(group)

    for i in constraint_equal:
        for j in range(len(i[0])):
            group = []
            constrained_charges.append(0)
            group.append(-i[0][j])
            group.append(i[1][j])
            constrained_indices.append(group)

    for i in constraint_groups:
        for j in range(1, len(i)):
            group = []
            constrained_charges.append(0)
            group.append(-i[j-1])
            group.append(i[j])
            constrained_indices.append(group)
    return constrained_charges, constrained_indices

def intermolecular_constraints(constraint_charge, constraint_equal):
    """Extracts intermolecular constraints from user constraint input

    Parameters
    ----------
    constraint_charge : list 
        list of list of lists of charges and atom indices list
        e.g. [[1, [[1, [1, 2]], [2, [3, 4]]]]]
        The sum of charges on atoms 1 and 2 of molecule 1
        and atoms 3 and 4 of molecule 2 will equal 1.
    constraint_equal : list
        list of list of list of indices of atoms to have
        equal charge in two molecules.
        e.g. [[[1, [1, 2]], [2, [3, 4]]]]
        charges on atoms 1 and 2 in molecule 1 will equal
        charges on  atoms 3 and 4 in molecule 2, respectively.

    Returns
    -------
    constrained_charges : list
        list of fixed charges 
    constrained_indices : list 
        list of lists of indices of atoms in a constraint
        negative number before an index means
        the charge of that atom will be subtracted.
    molecules : list
        list of lists of constrained molecules.

    Note
    ----
    Atom indices starts with 1 not 0
    """
    constrained_charges = []
    constrained_indices = []
    molecules = []
    for i in constraint_charge:
        constrained_charges.append(i[0])
        mol = []
        group_big = []
        for j in i[1]:
            mol.append(j[0])
            group = []
            for k in j[1]:
                group.append(k)
            group_big.append(group)
        constrained_indices.append(group_big)
        molecules.append(mol)

    for i in constraint_equal:
        for j in range(len(i[0][1])):
            molecules.append([i[0][0], i[1][0]])
            group = []
            constrained_charges.append(0)
            group.append([-i[0][1][j]])
            group.append([i[1][1][j]])
            constrained_indices.append(group)
    return constrained_charges, constrained_indices, molecules


def fit(options, inter_constraint):
    """Performs ESP and RESP fits.

    Parameters
    ----------
    options : list
        list of dictionaries of fitting options and internal data
    inter_constraint : dict
        dictionary of user-defined intermolecular constraints.

    Returns
    -------
    qf : list
        list of np.arrays of fitted charges
    labelf : list
        list of strings of fitting methods i.e. ESP and RESP
    notes : list
        list of strings of notes on the fitting
    """
    rest = options[0]['RESTRAINT'] 
    n_mols = len(options)
    qf = []
    labelf = []
    notes = []
    invr, coordinates, n_constraint, symbols, n_atoms = [], [], [], [], []
    constrained_charges, constrained_indices = [], []
    ndim = 0
    con_charges_sys, con_indices_sys, con_mol_sys = intermolecular_constraints(
                                                     inter_constraint['CHARGE'],
                                                     inter_constraint['EQUAL'])
    n_sys_constraint = len(con_charges_sys)
    for mol in range(n_mols):
        invr.append(options[mol]['invr'])
        coordinates.append(options[mol]['coordinates'])
        symbols.append(options[mol]['symbols'])
        n_atoms.append(len(symbols[mol]))
        constraint_charge = options[mol]['CONSTRAINT_CHARGE']
        constraint_equal = options[mol]['CONSTRAINT_EQUAL']
        constraint_groups = options[mol]['CONSTRAINT_GROUP']
        # Get user-defined constraints
        charges, indices = intramolecular_constraints(constraint_charge,
                                                       constraint_equal,
                                                       constraint_groups)
        constrained_charges.append(charges)
        constrained_indices.append(indices)
        n_con = len(charges)
        if mol == 0:
            n_con += 1
        n_constraint.append(n_con)
        ndim += n_atoms[mol] + n_constraint[mol]
    n_atoms = np.array(n_atoms)
    n_constraint = np.array(n_constraint)
    symbols = np.array(symbols, dtype='str')
    # Additional constraint to make charges in different molecules equal
    # to the charge in the first molecule
    # Also, Total charges = molecular charge
    ndim += n_sys_constraint
    a = np.zeros((ndim, ndim))
    b = np.zeros(ndim)
    
    edges_i = 0
    edges_f = 0
    indices = []
    # Bayly:93:10271 (Eqs. 12-14)
    for mol in range(n_mols):
        indices.append(range(edges_i, edges_i+n_atoms[mol]))
        # Construct A: A_jk = sum_i [(1/r_ij)*(1/r_ik)]
        inv = invr[mol].reshape((1, invr[mol].shape[0], invr[mol].shape[1]))
        a[edges_i:n_atoms[mol]+edges_i,
          edges_i:n_atoms[mol]+edges_i] = np.einsum("iwj, iwk -> jk", inv, inv)

        # Construct B: B_j = sum_i (V_i/r_ij)
        b[edges_i:n_atoms[mol]+edges_i] = np.dot(options[mol]['esp_values'], invr[mol])
        # Sum of point charges = molecular charge
        a[edges_i:n_atoms[mol]+edges_i, edges_i:n_atoms[mol]+edges_i] *= options[mol]['WEIGHT']**2
        b[edges_i:n_atoms[mol]+edges_i] *= options[mol]['WEIGHT']**2
        edges_f += n_atoms[mol]
        if mol == 0:
            a[:n_atoms[0], n_atoms[0]] = 1
            a[n_atoms[0], :n_atoms[0]] = 1
            b[n_atoms[0]] = options[0]['mol_charge']
            edges_f += 1

        # Add constraints to matrices A and B
        for i in range(1, n_constraint[mol]+1):
            if mol == 0 and i == n_constraint[mol]:
                # To account for the total charge constraints in the first molecule
                break
            b[edges_f] = constrained_charges[mol][i-1]
            for k in constrained_indices[mol][i-1]:
                if k > 0:
                    a[edges_f, edges_i+k-1] = 1
                    a[edges_i+k-1, edges_f] = 1
                else:
                    a[edges_f, edges_i-k-1] = -1
                    a[edges_i-k-1, edges_f] = -1
            edges_f += 1
        edges_i = edges_f
    indices = np.array(indices).flatten()

        # Add intermolecular constraints to A and B
    
    if n_mols > 1:
        for i in range(n_sys_constraint):
            b[edges_f] = con_charges_sys[i]
            for k in range(len(con_indices_sys[i])):
                for l in con_indices_sys[i][k]:
                    index = con_mol_sys[i][k]-1
                    index = int(np.sum(n_atoms[:index]) + np.sum(n_constraint[:index]))
                    if l > 0:
                        a[edges_f, index+l-1] = 1
                        a[index+l-1, edges_f] = 1
                    else:
                        a[edges_f, index-l-1] = -1
                        a[index-l-1, edges_f] = -1
            edges_f += 1

    labelf.append('ESP')
    q, note = esp_solve(a, b)
    qf.append(q[indices])
    notes.append(note)
    if not rest:
        return qf, labelf, notes
    else:
        ihfree, resp_a, resp_b = [], [], []
        for mol in range(n_mols):
            ihfree.append(options[mol]['IHFREE'])
            resp_a.append(options[mol]['RESP_A'])
            resp_b.append(options[mol]['RESP_B'])
        toler = options[0]['TOLER']
        maxit = options[0]['MAX_IT']
        # Restrained ESP 
        labelf.append('RESP')
        q, note = iterate(q, a, b, resp_a, resp_b, ihfree, symbols, toler, maxit,
                    n_atoms, n_constraint, indices)
        qf.append(q)
        notes.append(note)
        return qf, labelf, notes
