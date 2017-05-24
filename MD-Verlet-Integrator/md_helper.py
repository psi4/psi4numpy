# Helper "Library" to compute Molecular Dynamics (MD) Trajectories
# Numpy and OS python modules are required
#
# MD and Velocity Verlet Algorithms were taken from: :
# Computational Soft Matter:  From Synthetic Polymers to Proteins, Lecture Notes,
# Norbert Attig, Kurt Binder, Helmut Grubmuller, Kurt Kremer (Eds.),
# John von Neumann Institute for Computing, Julich,
# NIC Series, Vol. 23, ISBN 3-00-012641-4, pp. 1-28, 2004
#
# Created by: Leonardo dos Anjos Cunha
# Date: 05/16/17
# License: GPL v3.0

import psi4
import numpy as np
import os

# Function: md_trajectories
# Create .xyz file with MD Trajectories for all points
# Arguments: 
#       max_md_steps (integer): Number of MD steps calculated
def md_trajectories(max_md_step):
    with open('md_trajectories.xyz','w') as outfile:
      for i in range(1,max_md_step+1):
        with open('md_step_'+str(i)+'.xyz')as infile:
            outfile.write(infile.read())
    os.system('rm md_step*')

# Function: integrator
# Select the type of integration algorithm to propagate the trajectories
# Only velocity Verlet implemented (date: 05/23/17 - LAC)
# Arguments:
#       int_alg (string): Integrator Algorith to be used
#           a) veloc_verlet: Use Velocity Verlet Algorithm
#       timestep (float): Time step to be used on the MD trajectory propagation step
#       pos (array (natom,3)): Array with old positions/trajectories
#       veloc (array (natom,3)): Array with old velocities
#       accel (array (natom,3)): Array with old accelerations
#       molec (Psi4 molecule): Psi4 molecule object to be used
#       grad_method (string): Method to be used to calculate energies and forces
def integrator(int_alg,timestep,pos,veloc,accel,molec,grad_method):
    natoms = molec.natom()
    atom_mass = np.asarray([molec.mass(atom) for atom in range(natoms)])*amu2au

    # Velocity Verlet Integrator
    if int_alg=='veloc_verlet':
        vel_new =  veloc+0.5*timestep*accel
        pos_new =  pos+timestep*vel_new
        molec.set_geometry(psi4.core.Matrix.from_array(pos_new))
        E,force_new = get_forces(grad_method)
        accel_new = force_new/(atom_mass.reshape((natoms,1)))
        vel_new += 0*5*timestep*accel_new
    return pos_new,vel_new,accel_new,E

# Function: get_forces
# Select the method (QC or FF) to be used to calculate energies and forces for the system
# Only Psi4 supported methods implemented (date: 05/23/17 - LAC)
# Arguments:
#       grad_method (string): Method to be used to calculate forces and energies
def get_forces(grad_method):
    E,wfn = psi4.energy(grad_method,return_wfn=True)
    force = -np.asarray(psi4.gradient(grad_method,ref_wfn=wfn))
    return E,force

