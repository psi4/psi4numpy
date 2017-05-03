import numpy as np
import psi4
import os

psi4.core.set_output_file('md_output.dat',False)

molec = psi4.geometry("""
H 0 0 0
H 0 0 1
""")

# Global Constants (Atomic Units conversion)
fs_timeau = 41.34137314
amu2au = 1822.8884850

#MD Options
timestep =  5                       # Time step for each iteration in fs
max_md_step = 10000                 # Number of MD iterations
veloc0 = np.zeros((2,3))            # Numpy array (natoms,3) with inital velocities
trajec = True                       # Boolean: Save all trajectories in a single xyz file (md_trajectories)  for visualization
grad_method = 'hf/3-21g'            # QC method (with basis set) for energy and gradient
opt    = False                      # Optimize geometry using F=ma

# Molecular Dynamics Function
def molec_dyn(timestep,max_md_step,veloc0,grad_method,trajec = True,opt=False):
    
    #Get initial positions,velocities, accelerations and forces
    #timestep *= fs_timeau
    energy,wfn = psi4.energy(grad_method,return_wfn=True)
    forces = -np.asarray(psi4.gradient(grad_method,ref_wfn=wfn))
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
        pos_new,vel_new,accel_new,energy_new = veloc_verlet(timestep,pos,veloc,accel,atom_mass,natoms,grad_method)
        pos = pos_new
        if (not opt):
            veloc = vel_new
        accel = accel_new
        energy = energy_new
    md_energy.close()
    if trajec:
        md_trajectories(max_md_step)
    print "Done with Molecular Dynamics Program!"

# Velocity Verlet Integrator
def veloc_verlet(timestep,pos,vel,accel,atom_mass,natoms,grad_method):
    vel_new =  vel+0.5*timestep*accel
    pos_new =  pos+timestep*vel_new
    molec.set_geometry(psi4.core.Matrix.from_array(pos_new))
    E,wfn = psi4.energy(grad_method,return_wfn=True)
    force_new = -np.asarray(psi4.gradient(grad_method,ref_wfn=wfn))
    accel_new = force_new/(atom_mass.reshape((natoms,1)))
    vel_new += 0*5*timestep*accel_new
    return pos_new,vel_new,accel_new,E

def md_trajectories(max_md_step):
    with open('md_trajectories.xyz','w') as outfile:
      for i in range(1,max_md_step+1):
        with open('md_step_'+str(i)+'.xyz')as infile:
            outfile.write(infile.read())
    os.system('rm md_step*')

molec_dyn(timestep,max_md_step,veloc0,grad_method,trajec,opt)
