from renu import pdb
import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Bio import PDB
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
import pyrosetta as py
py.init("-mute all")


### functions to find the coordinate values

def read_pdb_coord(input_pdb):
    """
    This function uses biopython package for consistency
    function that reads a pdb file and outputs the xyz coordinates of the atoms in it
    output is a tuple of this form : (x, y,bz) where each coord is a list with all the coordinates
    """
    x_coord = []
    y_coord = []
    z_coord = []

    parser = PDB.PDBParser(QUIET=True)
    io = PDB.PDBIO()
    struct = parser.get_structure('easter_egg', input_pdb)

    for model in struct:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    x,y,z = atom.get_coord()
                    
                    # Append the coordinates to the lists
                    x_coord.append(x)
                    y_coord.append(y)
                    z_coord.append(z)
    return (x_coord, y_coord, z_coord)

def read_pdb_coord_window(input_pdb, z_height, window_size=10):
    """
    This function uses biopython package for consistency
    function that reads a pdb file and outputs the xyz coordinates of the atoms in it
    output is a tuple of this form : (x, y,bz) where each coord is a list with all the coordinates
    """
    x_coord = []
    y_coord = []
    z_coord = []

    parser = PDB.PDBParser(QUIET=True)
    io = PDB.PDBIO()
    struct = parser.get_structure('easter_egg', input_pdb)

    for model in struct:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    x,y,z = atom.get_coord()
                    if z > z_height-(window_size/2) and z < z_height+(window_size/2):
                        if atom.name in ["N", "CA", "C1"]:
                            # Append the coordinates to the lists
                            x_coord.append(x)
                            y_coord.append(y)
                            z_coord.append(z)
    return (x_coord, y_coord, z_coord)
    

def find_z1(z):
    """
    function for when I remove one of the md membrane to only look for one peak
    """
    # Compute KDE using scipy's gaussian_kde
    kde = gaussian_kde(z)
    z_values = np.linspace(min(z), max(z), 1000)  # Define x range for KDE plot
    density_values = kde(z_values)  # Evaluate the KDE on the x range

    # Use find_peaks to identify the peaks in the density values
    peaks, _ = find_peaks(density_values)
    # Output the peak locations for reference
    peak_locations = z_values[peaks]
    return peak_locations[0]


def find_z1_and_z2(z):
    """
    Find at which values of z the bilayers are found
    """
    # Compute KDE using scipy's gaussian_kde
    kde = gaussian_kde(z)
    z_values = np.linspace(min(z), max(z), 1000)  # Define x range for KDE plot
    density_values = kde(z_values)  # Evaluate the KDE on the x range

    # Use find_peaks to identify the peaks in the density values
    peaks, _ = find_peaks(density_values)
    # Output the peak locations for reference
    peak_locations = z_values[peaks]
    z1 = peak_locations[0]
    z2 = peak_locations[1]
    mid_z = (z1 + z2)/2
    return z1, z2, mid_z

def find_x_data(x):
    """
    returns the the value on x where we can find borders of the TMB as well as the middle point 
    """
    # Compute KDE using scipy's gaussian_kde
    kde = gaussian_kde(x)
    x_values = np.linspace(min(x), max(x), 1000)  # Define x range for KDE plot
    density_values = kde(x_values)  # Evaluate the KDE on the x range

    # Use find_peaks to identify the peaks in the density values
    peaks, _ = find_peaks(density_values)

    # Output the peak locations for reference
    peak_locations = x_values[peaks]
    x1 = peak_locations[0]
    x2 = peak_locations[-1]
    mid_x = (x1 + x2)/2
    return x1, x2, mid_x

def find_y_data(y):
    # Compute KDE using scipy's gaussian_kde
    kde = gaussian_kde(y)
    y_values = np.linspace(min(y), max(y), 1000)  # Define x range for KDE plot
    density_values = kde(y_values)  # Evaluate the KDE on the x range

    # Use find_peaks to identify the peaks in the density values
    peaks, _ = find_peaks(density_values)

    # Output the peak locations for reference
    peak_locations = y_values[peaks]
    y1 = peak_locations[0]
    y2 = peak_locations[-1]
    mid_y = (y1 + y2)/2
    return y1, y2, mid_y


def get_radius_pore_centers(input_tmb, z_height):
    """
    returns, radius, cicrle center x and circle center y"""
    x, y, z = read_pdb_coord_window(input_tmb, z_height) 
    data_x = find_x_data(x)
    data_y = find_y_data(y)
    pore_center_x = data_x[2]
    pore_center_y = data_y[2]
    radius = ((data_x[1]-data_x[0]) + (data_y[1] - data_y[0]))/2
    return radius, pore_center_x, pore_center_y

def get_x_y_z_coord(pore_center_x, pore_center_y, num_points_x, num_points_y, z1, z2, spacing, halo, radius, bilayer=False):
    """ Function that returns x, y, and z coordinates as lists. """

    # Circle parameters
    # the number of residues we want in our virtual_residues halo to be
    circle_outer_radius = radius + (halo*spacing)

    x_coord = []
    y_coord = []
    z_coord = []


    if bilayer:
        # Generate coordinates for each sheet layer
        for z in [z1, z2]:  # Two layers, one at z1 the other at z2
            for i in range(-num_points_x, num_points_x):
                for j in range(-num_points_y, num_points_y):
                    x = i * spacing - (halo*spacing)
                    y = j * spacing - (halo*spacing)

                    # Check if the point is in the circle surface
                    distance_from_center = np.sqrt((x - pore_center_x)**2 + (y - pore_center_y)**2)
                    if distance_from_center > radius and distance_from_center < circle_outer_radius:
                        x_coord.append(x)
                        y_coord.append(y)
                        z_coord.append(z)
    else: # if bilayer is false only create one layer of virtual residues
        for z in [z1]:
            for i in range(-num_points_x, num_points_x):
                for j in range(-num_points_y, num_points_y):
                    x = i * spacing - (halo*spacing)
                    y = j * spacing - (halo*spacing)

                    # Check if the point is in the circle surface
                    distance_from_center = np.sqrt((x - pore_center_x)**2 + (y - pore_center_y)**2)
                    if distance_from_center > radius and distance_from_center < circle_outer_radius:
                        x_coord.append(x)
                        y_coord.append(y)
                        z_coord.append(z)
    return x_coord, y_coord, z_coord


def get_num_xy_points(radius, spacing, halo, extra):
    """returns the number of points needed on the x and y axis for our planes"""
    c_2 = 2*radius + 2*halo*spacing + extra*2
    return int(round(2*c_2/spacing))



def create_virtual_residues(pore_center_x, pore_center_y, z_height, spacing, halo, radius, bilayer=False):
    """
    z_input is the height where I will form my layer of vr
    Function creates a layer of virtual residues (unknown in this case(basically an alanine)) of modulable sizes (inputs)*
    num_points = how many points for the x * y grid
    spacing = how big the distance between the points should be
    thickness = distance between the two bilayers
    out_pdb = name of the out put pdb file (needs .pdb) at the end of it
    halo = the halo's thickness
    Radius = the radius of the pore
    """  

    if bilayer:
        z_list = find_z1_and_z2(z_height)
        z1, z2 = z_list[0], z_list[1]
    if not bilayer:
        z1 = z_height
        z2 = None


    num_points_x = get_num_xy_points(radius, spacing, halo, 3)
    num_points_y = num_points_x
    # set the coordinates in our bilayer,
    x_coord, y_coord, z_coord = get_x_y_z_coord(pore_center_x, pore_center_y, num_points_x=num_points_x, num_points_y=num_points_y, z1=z1, z2=z2,  spacing=spacing, radius=radius, halo=halo, bilayer=bilayer)


    # create a G residue so that we have the coordinates of the atoms of the Glycine in regards to the N atom whose coordinates are (0, 0, 0)
    model = py.pose_from_sequence("A")
    res_model = model.residue(1)
    all_atom_names = ["N", "CA", "C", "O", "OXT", "CB", "1H", "2H", "3H", "HA",  "1HB", "2HB", "3HB"]

    # this paragraph creates a pdb files using the pose object from pyrosetta, 

    # test is a pose with all the residues we need for our bilayer 
    # maybe the bug is in here (almost sure that it is, try to replace x_coord with z_coord, strange results) ????
    test = py.pose_from_sequence("A" * len(x_coord))

    for index_residue, (x, y, z) in enumerate(zip(x_coord, y_coord, z_coord), start=1): # iterate over every glycine in our file
        for index_atom in range(1, len(test.residue(index_residue).atoms())+1): # iterate over every atom in our picked glycine

            test.residue(index_residue).set_xyz(index_atom,
                                        py.rosetta.numeric.xyzVector_double_t(
                                            res_model.xyz(all_atom_names[index_atom-1]).x + x,
                                                res_model.xyz(all_atom_names[index_atom-1]).y + y,
                                                    res_model.xyz(all_atom_names[index_atom-1]).z + z))
    
    
    return test

def find_start_line_of_last_chain(pdb_file):
    chain_order = []
    atom_lines = []

    with open(pdb_file, 'r') as f:
        for idx, line in enumerate(f, 1):
            if line.startswith("ATOM") or line.startswith("HETATM"):
                chain_id = line[21]
                atom_lines.append((idx, chain_id))
                if not chain_order or chain_id != chain_order[-1]:
                    chain_order.append(chain_id)


    last_chain = chain_order[-1]

    for line_num, chain_id in atom_lines:
        if chain_id == last_chain:
            return line_num

    return None


def replace_ala_with_unk(input_pdb, output_pdb):
    """
    Updated version of the below commented one"""
    start_line = find_start_line_of_last_chain(input_pdb)

    with open(input_pdb, 'r') as file:
        lines = file.readlines()

    for i in range(start_line - 1, len(lines)):
        lines[i] = lines[i].replace(" ALA ", " UNK ")

    with open(output_pdb, 'w') as file:
        file.writelines(lines)

# def replace_ala_with_unk(input_pdb, output_pdb, start_line):

#     # Read the file
#     with open(input_pdb, 'r') as file:
#         lines = file.readlines()

#     for i in range(start_line - 1, len(lines)):
#         lines[i] = lines[i].replace(" ALA ", " UNK ")

#     with open(output_pdb, 'w') as file:
#         file.writelines(lines)


def merge_bilayers_and_TMB(input_tmb_pdb, vr_membrane_pdb):
    """"
    tmb_pdb = fichier pdb avec le barrel
    md_membrane_pdb = fichier pdb avec la membrane
    extension_membrane_pdb = fichier pdb avec l'extension de la membrane

    explications: à la fin le tmb_pdb ser une pose avec le tmb, le bilayer md, et l'extension de la bilayer
    """
    tmb_pose = py.pose_from_pdb(input_tmb_pdb)
    vr_pose = vr_membrane_pdb

    py.rosetta.core.pose.append_pose_to_pose(tmb_pose, vr_pose)
    return tmb_pose


def main(input_tmb_pdb, output_pdb, spacing, halo, z_height):
    data = get_radius_pore_centers(input_tmb_pdb, z_height)
    radius = data[0]
    pore_center_x = data[1]
    pore_center_y = data[2]
    virtual_residues_pose = create_virtual_residues(z_height=z_height, pore_center_x=pore_center_x, pore_center_y=pore_center_y, 
                                                    spacing=spacing, halo=halo, radius=radius, bilayer=False)
    merged_pose = merge_bilayers_and_TMB(input_tmb_pdb, virtual_residues_pose)
    if halo < 10:
        merged_pose.dump_pdb(output_pdb)
    else:
        merged_pose.dump_pdb(output_pdb)
    replace_ala_with_unk(output_pdb, output_pdb)

from transform_diluted_2_virtual_residues import translate_vr




main(input_tmb_pdb="/home/deider/memoire/experiences/project_ppm/thomas_barrel_cleaned.pdb",
    output_pdb="/home/deider/memoire/experiences/thomas_barrel_plus_VR3.pdb",
    spacing=7,
    halo=7,
    z_height=77     # Modify this value to the value that you found at step 2
)
