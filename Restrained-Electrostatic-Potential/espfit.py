import numpy as np
import copy

def esp_solve(a, b):
    """Function to solve for point charges: A*q = B
       input:  a: array of matrix A
               b: array of matrix B
       output: q: array of charges
    """
    q = np.linalg.solve(a, b)
    # Warning for near singular matrix
    # in case np.linalg.solve does not detect singularity
    note = ''
    if np.linalg.cond(a) > 1/np.finfo(a.dtype).eps:
        note = "Possible fit problem; singular matrix"
    return q, note

def restraint(q, akeep, resp_a, resp_b, ihfree, symbols, n_atoms, n_constraints):
    """Function to add hyperbolic restraint to matrix A
       input: 
            q: array of charges
            akeep: array of unrestrained A matrix
            resp_a: float; restraint scale a
            resp_b: float; restraint parabola tightness b
            ihfree: bool: hydrogen excluded or included in restraint
            symbols: string array of element symbols
            n_atoms: list of the number of atoms in each molecule
            n_constraints: list of the number of constraints for each molecule
       output:
            a: restrained A array
    """

    # hyperbolic Restraint
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
            maxit, n_atoms, n_constraints, indecies):
    """Function to iterate the RESP fitting procedure
       input:
                q: array of initial charges
                akeep: array of unrestrained A matrix
                b: array of matrix B
                resp_a: float; restraint scale a
                resp_b: float; restraint parabola tightness b
                ihfree: bool: hydrogen excluded or included in restraint
                symbols: array of element symbols
                toler: float; tolerance for charges in the fitting
                maxit: int; maximum number of iterations
                n_atoms: list of the number of atoms in each molecule
                n_constraints: list of the number of constraints for
                                each molecule
                indecies: an array of the indecies for the atoms in the
                          A and B matrices
        output:
                q: array of fitted charges
    """
    n_mols = len(n_atoms)
    qkeep = q[indecies]
    niter = 0
    difm = 1
    while difm > toler and niter < maxit:
        index = 0
        niter += 1
        a = restraint(q[indecies], akeep, resp_a, resp_b, ihfree,\
                      symbols, n_atoms, n_constraints)
        q, note = esp_solve(a, b)
        q_q = q[indecies]
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
    """Function to extract intramolecular constraints from user constraint input
       input:
             constraint_charge: list of lists of charges and atom indecies list
                               e.g. [[0, [1, 2]], [1, [3, 4]]]
                               The sum of charges on 1 and 2 will equal 0
                               The sum of charges on 3 and 4 will equal 1
             constraint_equal: list of lists of two lists of indecies of atoms to 
                               have equal charge element by element
                               e.g. [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
                               atoms 1 and 3 will have equal charge
                               atoms 2 and 4 will have equal charge
                               and similarly for 5, 6, 7 and 8
             constraint_group: list of lists of indecies of atoms to have equal charge
                              e.g. [[1, 2], [3, 4]]
                              atoms 1 and 2 will have equal charge
                              atoms 3 and 4 will have equal charge
       output:
             constrained_charges: list of fixed charges 
             constrained_indecies: list of lists of indices of atoms in a constraint
                                   negative number before an index means
                                   the charge of that atom will be subtracted.
       Notes: Atom indecies starts with 1 not 0.
              Total charge constraint is added by default for the first molecule.
    """
                        
    constrained_charges = []
    constrained_indecies = []
    for i in constraint_charge:
        constrained_charges.append(i[0])
        group = []
        for k in i[1]:
            group.append(k)
        constrained_indecies.append(group)

    for i in constraint_equal:
        for j in range(len(i[0])):
            group = []
            constrained_charges.append(0)
            group.append(-i[0][j])
            group.append(i[1][j])
            constrained_indecies.append(group)

    for i in constraint_groups:
        for j in range(1, len(i)):
            group = []
            constrained_charges.append(0)
            group.append(-i[j-1])
            group.append(i[j])
            constrained_indecies.append(group)
    return constrained_charges, constrained_indecies

def intermolecular_constraints(constraint_charge, constraint_equal):
    """Function to extract intermolecular constraints from user constraint input
      input:
            constraint_charge: list of list of lists of charges and atom indecies list
                              e.g. [[1, [[1, [1, 2]], [2, [3, 4]]]]]
                              The sum of charges on atoms 1 and 2 of molecule 1
                              and atoms 3 and 4 of molecule 2 will equal 1.
            constraint_equal: list of list of list of indecies of atoms to have
                              equal charge in two molecules.
                              e.g. [[[1, [1, 2]], [2, [3, 4]]]]
                              charges on atoms 1 and 2 in molecule 1 will equal
                              charges on  atoms 3 and 4 in molecule 2, respectively.
            output:
            constrained_charges: list of fixed charges 
            constrained_indecies: list of lists of indices of atoms in a constraint
                                  negative number before an index means
                                  the charge of that atom will be subtracted.
            molecules: list of lists of constrained molecules.
      Note: Atom indecies starts with 1 not 0
    """
    
    constrained_charges = []
    constrained_indecies = []
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
        constrained_indecies.append(group_big)
        molecules.append(mol)

    for i in constraint_equal:
        for j in range(len(i[0][1])):
            molecules.append([i[0][0], i[1][0]])
            group = []
            constrained_charges.append(0)
            group.append([-i[0][1][j]])
            group.append([i[1][1][j]])
            constrained_indecies.append(group)
    return constrained_charges, constrained_indecies, molecules


def fit(options, inter_constraint):
    """Function to perform ESP and RESP fits.
       input:
            options: dictionary of fitting options and internal data
            inter_constraint: dictionary of user-defined intermolecular
                              constraints.
       output:
            qf: list of arrays of fitted charges
            labelf: list of strings of fitting methods i.e. ESP and RESP
    """

    rest = options[0]['RESTRAINT'] 
    n_mols = len(options)
    qf = []
    labelf = []
    notes = []
    invr, coordinates, n_constraint, symbols, n_atoms = [], [], [], [], []
    constrained_charges, constrained_indecies = [], []
    ndim = 0
    con_charges_sys, con_indecies_sys, con_mol_sys = intermolecular_constraints(
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
        charges, indecies = intramolecular_constraints(constraint_charge,
                                                       constraint_equal,
                                                       constraint_groups)
        constrained_charges.append(charges)
        constrained_indecies.append(indecies)
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
    indecies = []
    for mol in range(n_mols):
        indecies.append(range(edges_i, edges_i+n_atoms[mol]))
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
            for k in constrained_indecies[mol][i-1]:
                if k > 0:
                    a[edges_f, edges_i+k-1] = 1
                    a[edges_i+k-1, edges_f] = 1
                else:
                    a[edges_f, edges_i-k-1] = -1
                    a[edges_i-k-1, edges_f] = -1
            edges_f += 1
        edges_i = edges_f
    indecies = np.array(indecies).flatten()

        # Add intermolecular constraints to A and B
    
    if n_mols > 1:
        for i in range(n_sys_constraint):
            b[edges_f] = con_charges_sys[i]
            for k in range(len(con_indecies_sys[i])):
                for l in con_indecies_sys[i][k]:
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
    qf.append(q[indecies])
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
                    n_atoms, n_constraint, indecies)
        qf.append(q)
        notes.append(note)
        return qf, labelf, notes
