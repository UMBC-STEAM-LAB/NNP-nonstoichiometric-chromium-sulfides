#!/usr/bin/env python
# coding: utf-8

import os, sys, math, time, random
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from lammps import lammps, PyLammps
from mpi4py import MPI
from ase.io import read, write, lammpsdata
from ase.units import kB
from ase.geometry import cell_to_cellpar
from ase import Atoms
from ase.build.tools import sort
from pymatgen.core.structure import Structure
from pymatgen.io.vasp import Poscar
from pymatgen.io.ase import AseAtomsAdaptor
import atomman as am
from pymatgen.core.periodic_table import Element, DummySpecies
from pymatgen.analysis.local_env import VoronoiNN
from ovito.io.ase import ase_to_ovito, ovito_to_ase
from ovito.pipeline import StaticSource, Pipeline
from ovito.modifiers import WignerSeitzAnalysisModifier

# Initialize MPI and get rank and size
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

################################################################
# Inputs part
################################################################

# Set a fixed seed for the random number generator in each process
random.seed(42)

# Set initial temperature and number of Monte Carlo steps
vac_conc = 7/15
temp = 300.0
n_mc_steps = 500000
p_hmc = 0.88
randomize = True
strain_xy = -0.075

# Read primitive cell
struc_prim = read('POSCAR_slab')


# Make supercell
struc_super = AseAtomsAdaptor.get_structure(struc_prim)
struc_super.make_supercell([16,9,1])
struc_super = AseAtomsAdaptor.get_atoms(struc_super)

# Get Cr sublattice indices
Cr_sublattice_indices = []
for i, at in enumerate(struc_super):
    if at.symbol=='Cr':
        Cr_sublattice_indices.append(i)
        
        
# Generate a random initial distribution of vacancies 
if rank == 0:
    N_vac = int(len(Cr_sublattice_indices) * vac_conc)
    random_sequence = np.random.permutation(Cr_sublattice_indices)
    vac_indices = random_sequence[:N_vac]
else:
    vac_indices = None

# Broadcast the random sequence to all other ranks
vac_indices = comm.bcast(vac_indices, root=0)


# Decorate the structure with Cr vacancy defects
struc_defect = struc_super.copy()
occup = struc_defect.get_chemical_symbols()
for i in vac_indices:
    occup[i] = 'X'
struc_defect.set_chemical_symbols(occup)
    
# Apply strain on lattice parameters 
strained_structure = struc_defect.copy()

# Apply the strain to the current lattice parameters
cellpars = strained_structure.get_cell()
cellpars[0] *= 1 + strain_xy
cellpars[1] *= 1 + strain_xy
# Set the new cell parameters
strained_structure.set_cell(cellpars, scale_atoms=True)

# define the initial strucuture
initial_structure = strained_structure.copy()

N_atoms = len([i for i in initial_structure if (i.symbol == 'Cr' or i.symbol == 'S')])

# Get the distance matrix for all sites
# Load the array from the file
distance_matrix = np.load('../distance_matrix_n.npy')

################################################################
# Nearest neighbor finder part
################################################################
# find nearest neighbor Cr atoms to a chose vacancy site
def get_nn_indices(structure, chosen_vacancy, distance_matrix):
    
    # Define the cutoff distance (replace with your own cutoff)
    cutoff_distance = 4.5 + 0.4  # Cr-Cr bond length in CrS2 (4.5 Å) + skin&cell_variation distance (0.4 Å)
    # Build a list of the nearest neighbor atoms within the cutoff distance from X atom
    nn_indices = []
    for i in range(len(distance_matrix)):
        if np.linalg.norm(distance_matrix[chosen_vacancy][i]) <= cutoff_distance and structure[i].symbol == 'Cr':
            nn_indices.append(i)
    
    return nn_indices


################################################################
# LAMMPS calculator
################################################################

def lammps_energy_minimization(real_structure, mc_step): 
    
    file_i = 'file-{}_i.dat'.format(str(mc_step))
    file_i_dump = './data_files/file-{}_i.dat'.format(str(mc_step))
    file_f_dump = './data_files/file-{}_f.dat'.format(str(mc_step))   
                                    
    # Dump LAMMPS initial data file
    write(file_i, real_structure, 'lammps-data')
    
    if mc_step %25 == 0:
        write(file_i_dump, real_structure, 'lammps-data')
    
    # Synchronize all ranks 
    comm.barrier()

    # Create lammps instance
    lmp = lammps()
    lmp.command("variable cfgFile string " + file_i)
    lmp.command("variable dt equal 0.001")
    lmp.command("variable nnpCutoff equal 6.5")
    lmp.command("variable nnpDir string '/home/akram/common/akram/CrS_project/SI/nnp_model/nnp-data_fisa3_iter_80'")
    lmp.command("variable mass_Cr equal  51.9961")
    lmp.command("variable mass_S equal 32.065")
    lmp.command("units metal")
    lmp.command("boundary p p p")
    lmp.command("atom_style atomic")
    lmp.command("box tilt large")
    lmp.command("read_data ${cfgFile}")
    lmp.command("mass 1 ${mass_Cr}")
    lmp.command("mass 2 ${mass_S}")
    lmp.command("pair_style nnp dir ${nnpDir} showew no showewsum 1 \
    resetew no maxew 500000000 cflength 1.00 cfenergy 1.00 emap '1:Cr,2:S'")
    lmp.command("pair_coeff * * ${nnpCutoff}")
    lmp.command("change_box all triclinic")
    # thermo outputs
    lmp.command("thermo 50")
    lmp.command("thermo_style custom step temp vol pxx pyy pzz pe")

    # Relax box / Minimize energy
    lmp.command("min_modify dmax 0.025")
    maxstep = int(np.ceil(80 * (1 - np.exp(-(mc_step+1) / (100000)))))
    lmp.command("minimize 1e-6 1e-5 "+ str(maxstep) + " " + str(maxstep))
    #######
    if mc_step %25 == 0:
        forces_dump_str = './forces_dump/forces_'+str(mc_step)+'.dump'
        lmp.command("dump dumpy all custom 1 "+ forces_dump_str +" id type x y z fx fy fz")
        lmp.command("dump_modify dumpy format float %.13e")
        lmp.command("run 0")
        # write final data
        lmp.command("write_data " + file_f_dump)
    
    # Synchronize all ranks 
    comm.barrier()
    

    #######
    # Synchronize all ranks 
    comm.barrier()
    
    # Get the final energy after minimization
    final_energy = lmp.extract_compute("thermo_pe",0,0) # extract value(s) from a compute
    
    # Synchronize all ranks 
    comm.barrier()
            
    if rank == 0:
        os.remove(file_i)                                

    return final_energy


################################################################
# MC part
################################################################

mc_steps = []
mc_accepted_moves = []
mc_energies = []
mc_structures = []
mc_cluster_sizes = []
mc_moves = []
chosen_vacs_hist = []
chosen_crs_hist = []

# Perform Monte Carlo simulation
structure = initial_structure.copy()
mc_step = 0
energy = 0

while mc_step < n_mc_steps:   
    Cr_indices = [i for i, at in enumerate(structure) if at.symbol=='Cr']
    vac_indices = [i for i, at in enumerate(structure) if at.symbol=='X']
    
    proposed_occupation = structure.get_chemical_symbols()
    proposed_structure = structure.copy()
    
    if mc_step > 0: # Avoid modifying the occupation of the 0th mc_step     
        if rank == 0:
            rnd_num = random.random()
        else:
            rnd_num = None
        rnd_num = comm.bcast(rnd_num, root=0)
        
        if(rnd_num <= p_hmc and mc_step>1000): # Atomic swapping ###  
            
            # Choose cluster size for the MC swap
            if rank == 0:
                if mc_step <= 6000:
                    cluster_size = random.randint(1, random.randint(1, 6)) # Cluster size
                if mc_step > 6000 and mc_step <= 11000:
                    cluster_size = random.randint(1, random.randint(1, 3)) # Cluster size  
                if mc_step > 11000:
                    cluster_size = random.randint(1, 2) # Cluster size
                    
            else:
                cluster_size = None       
            # Broadcast chosen_neighbor to all ranks 
            cluster_size = comm.bcast(cluster_size, root=0)
            
            nns_nonempty = False
            nns_unique = False
            while nns_nonempty == False:
                # Choose random vacancies
                if rank == 0:
                    chosen_vacs = random.sample(vac_indices, cluster_size)
                else:
                    chosen_vacs = None         
                # Broadcast to all ranks 
                chosen_vacs = comm.bcast(chosen_vacs, root=0)

                nn_Crs = [get_nn_indices(proposed_structure, chosen_vac, distance_matrix) for chosen_vac in chosen_vacs]

                # Check if all sub-lists are non-empty
                nns_nonempty = all(len(nn) > 0 for nn in nn_Crs)
                        
                if nns_nonempty:    
                    nns_unique_count = 0
                    # Check for nn_Crs unqiueness
                    for k, nn in enumerate(nn_Crs):
                        remaining_nn_Crs = nn_Crs[:k] + nn_Crs[k+1:]
                        flattened_remaining = [q for remain in remaining_nn_Crs for q in remain]
                        if all(x in flattened_remaining for x in nn):
                            nns_unique_count += 0
                            break
                        else:
                            nns_unique_count += 1
                    if nns_unique_count == len(nn_Crs):
                        nns_unique = True
                        
                    if nns_unique:
                        if rank == 0:
                            chosen_Crs = []
                            # Loop through each list in nn_Crs and select one element
                            for nn in nn_Crs:
                                # Make sure the selected element is unique
                                unique_element = random.choice([x for x in nn if x not in chosen_Crs])
                                chosen_Crs.append(unique_element)
                        else:
                            chosen_Crs = None
                        # Broadcast to all ranks 
                        chosen_Crs = comm.bcast(chosen_Crs, root=0)    


                        # Propose the MC swaps 
                        for j in range(cluster_size):    
                            proposed_occupation[chosen_Crs[j]] = 'X'; proposed_occupation[chosen_vacs[j]] = 'Cr'
                        proposed_structure.set_chemical_symbols(proposed_occupation)             

        else: # Volumetric perturpation ###
            
            # Propose MC volume change
            proposed_structure_py = AseAtomsAdaptor.get_structure(proposed_structure)
            if rank == 0:
                del_a = 0;del_b = 0;del_c = random.uniform(-0.001, 0.001)
            else:
                del_a = None; del_b = None; del_c = None
             # Broadcast to all ranks 
            del_a = comm.bcast(del_a, root=0); del_b = comm.bcast(del_b, root=0); del_c = comm.bcast(del_c, root=0)  
            
            # Apply the proposed strain    
            proposed_structure_py.apply_strain([del_a, del_b, del_c])
            # Get the proposed structure
            proposed_structure = AseAtomsAdaptor.get_atoms(proposed_structure_py)
    
    # Make a copy of the initial structure
    real_structure = proposed_structure.copy()
    # Remove dummy vacant sites from the structure
    del real_structure[[atom.index for atom in real_structure if atom.symbol=='X']]   
    
    # Synchronize all ranks 
    comm.barrier()
    
 
    # Calculate energy of the proposed atomic configuration
    energy_after_swap = lammps_energy_minimization(real_structure, mc_step)
    
    # Calculate volume of the old atomic configuration
    vol_old = structure.get_volume()
    # Calculate volume of the proposed atomic configuration
    vol_new = proposed_structure.get_volume()
    
    # Synchronize all ranks 
    comm.barrier()
        
    # Decide whether to accept or reject MC move based on Metropolis algorithm
    delta_e = energy_after_swap - energy
    
    # generate a random number on rank 0
    if rank == 0:
        random_number = random.random()
    else:
        random_number = None

    # broadcast the random number to all ranks
    random_number = comm.bcast(random_number, root=0)

    
    if delta_e <= 0 or random_number < np.exp(-delta_e / (kB * temp) + N_atoms * np.log(vol_new/vol_old)):
        # Perform swap 
        # update structure
        structure = proposed_structure.copy()
        # update energy
        energy = energy_after_swap   
        # save the accepted mc swap
        mc_accepted_moves.append(1)
        if mc_step > 0:
            if(rnd_num <= p_hmc and mc_step>1000):
                mc_moves.append('swap')
                chosen_vacs_hist.append(chosen_vacs)
                chosen_crs_hist.append(chosen_Crs)
            else:
                mc_moves.append('volume')
                chosen_vacs_hist.append([])
                chosen_crs_hist.append([])
        else:
            mc_moves.append('not_accepted')
            chosen_vacs_hist.append([])
            chosen_crs_hist.append([])
            
            
    else:
        # save the rejected mc swap
        mc_accepted_moves.append(0) 
        if mc_step > 0:
            if(rnd_num <= p_hmc and mc_step>1000):
                mc_moves.append('swap')
                chosen_vacs_hist.append([])
                chosen_crs_hist.append([])
            else:
                mc_moves.append('volume')
                chosen_vacs_hist.append([])
                chosen_crs_hist.append([])
        else:
            mc_moves.append('not_accepted')
            chosen_vacs_hist.append([])
            chosen_crs_hist.append([])
    
    # Synchronize all ranks 
    comm.barrier()
    
    
    # Collect MC step data
    mc_steps.append(mc_step)
    mc_energies.append(energy/N_atoms)
    mc_structures.append(structure)
    if mc_step == 0:
        cluster_size = 0
    mc_cluster_sizes.append(cluster_size)
    
  
    
    # Increase mc_steps by 1    
    mc_step += 1     
    
    if mc_step % 30000 == 0 and mc_step > 0:
        # Dump the structures into a trajectory file
        write("trajectory.xyz", mc_structures, format='extxyz')
        
        mc_data = {'Monte Carlo Step': mc_steps, 'Sampled Energy': mc_energies, 'Accepted/Rejected': mc_accepted_moves,
                   'Cluster_size': mc_cluster_sizes, 'chosen_vacs_hist': chosen_vacs_hist,
                   'chosen_crs_hist': chosen_crs_hist,
                   'mc_move_type': mc_moves}
        df = pd.DataFrame(mc_data)
        df.to_csv('monte_carlo_results.csv', index=False)

