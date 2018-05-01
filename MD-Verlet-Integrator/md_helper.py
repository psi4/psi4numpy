"""
Helper functions used to compute Molecular Dynamics (MD) trajectories

References:
Algorithms & equations taken from [Attig:2004].
"""

__authors__ = "Leonardo dos Anjod Cunha"
__credits__ = ["Leonardo dos Anjod Cunha"]

__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__   = "BSD-3-Clause"
__date__      = "2017-05-16"

import psi4
import numpy as np
import os

# Converting Atomic Mass Unit (amu) to atomic units (au)
amu2au = 1822.8884850

def md_trajectories(max_md_step):
    """ Creates single .xyz file with MD Trajectories with all points.
        
        Parameters:
        ----------
            int max_md_steps -- Number of MD steps calculated

        Returns:
        -------
            file md_trajectories.xyz -- File containing all the trajectory points for visualization
    """

    with open('md_trajectories.xyz','w') as outfile:
      for i in range(1,max_md_step+1):
        with open('md_step_'+str(i)+'.xyz')as infile:
            outfile.write(infile.read())
    os.system('rm md_step*')

def integrator(int_alg,timestep,pos,veloc,accel,molec,grad_method):
    """ Selects the type of integration algorithm to propagate the trajectories
        Only velocity Verlet implemented (date: 05/23/17 - LAC)

        Parameters:
        ----------
            string int_alg --  Integrator Algorith to be used
                Options:
                a) veloc_verlet --  Use Velocity Verlet Algorithm
            float timestep -- Time step to be used on the MD trajectory propagation step
            array pos --  Numpy array (natoms,3) with old positions/trajectories
            array veloc --  Numpy array (natoms,3) with old velocities
            array accel -- Numpy array (natoms,3) with old accelerations
            psi4_object molec --  Psi4 molecule object to be used
            string grad_method: Method to be used to calculate energies and forces

        Returns:
        -------
            array pos_new -- Numpy array (natoms,3) with the updated positions
            array vel_new -- Numpy array (natoms,3) with the updated velocities
            array accel_new -- Numpy array (natoms,3) with the updated accelerations
            float E -- Updated energy of the system
    """

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

def get_forces(grad_method):
    """ Selects the method (QC or FF) to be used to calculate energies and forces for the system
        Only Psi4 supported methods implemented (date: 05/23/17 - LAC)
    
        Parameters:
        ----------
            string grad_method -- Method to be used to calculate forces and energies

        Returns:
        -------
            float E -- Energy of the system
            array force -- Numpy array (natoms,3) containing the forces acting on each atom of the system
    """

    E,wfn = psi4.energy(grad_method,return_wfn=True)
    force = -np.asarray(psi4.gradient(grad_method,ref_wfn=wfn))
    return E,force

