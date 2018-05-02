"""
A simple Psi4 script to compute Molecular Dynamics Trajectories using
Velocity Verlet integration.

References:
Equations and algorithms taken from [Attig:2004].
"""

__authors__ = "Leonardo dos Anjod Cunha"
__credits__ = ["Leonardo dos Anjod Cunha"]

__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__   = "BSD-3-Clause"
__date__      = "2017-05-16"

import numpy as np
import psi4
import os
import sys
sys.path.append('./')
import md_helper

psi4.core.set_output_file('md_output.dat',False)

molec = psi4.geometry("""
H 0 0 0
H 0 0 1
""")

# Global Constants (Atomic Units conversion)
fs_timeau = 41.34137314
amu2au = 1822.8884850

#MD Options
timestep =  5                       # Time step for each iteration in time atomic units
max_md_step = 100                 # Number of MD iterations
veloc0 = np.zeros((2,3))            # Numpy array (natoms,3) with inital velocities
trajec = True                       # Boolean: Save all trajectories in a single xyz file (md_trajectories)  for visualization
grad_method = 'hf/3-21g'            # Method (QC with basis set) for energy and gradient
int_alg = 'veloc_verlet'            # Algorithm to use as integrator
opt    = False                      # Optimize geometry using F=ma

# Molecular Dynamics Main Program
    
#Get initial positions,velocities, accelerations and forces
energy,forces = md_helper.get_forces(grad_method)
geom = np.asarray(molec.geometry())
natoms = molec.natom()
atom_mass = np.asarray([molec.mass(atom) for atom in range(natoms)])*amu2au
if opt:
    veloc = np.zeros((natoms,3))
else:
    veloc = veloc0
accel = forces/(atom_mass.reshape((natoms,1)))

# MD LOOP
pos = geom
# Saving energies of each iteration on md_energy file
md_energy = open('md_energy.dat','w')
md_energy.write('File with the energy of each MD trajectory point\n\n')
md_energy.write('Trajectory Number\tEnergy (Hartree)\n')
md_energy.write('{0:>3d}\t\t\t{1:10.8f}\n'.format(1,energy))
for i in range(1,max_md_step+1):
    
    # Saving energies and trajectory points
    md_energy.write('{0:>3d}\t\t\t{1:10.8f}\n'.format(i,energy))
    if trajec:
        molec.save_xyz_file('md_step_'+str(i)+'.xyz',False)

    # Updating positions velocities and accelerations using Velocity Verlet Integrator
    pos_new,vel_new,accel_new,energy_new = md_helper.integrator(int_alg,timestep,pos,veloc,accel,molec,grad_method)
    pos = pos_new
    if (not opt):
        veloc = vel_new
    accel = accel_new
    energy = energy_new
md_energy.close()
if trajec:
    md_helper.md_trajectories(max_md_step)
print "Done with Molecular Dynamics Program!"
