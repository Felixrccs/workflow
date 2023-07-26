import re, os, random, sys

import numpy as np
import pandas as pd
import datetime as dt

from ase.data import atomic_masses, chemical_symbols
from ase.io import read
from ase.io.lammpsdata import write_lammps_data
from ase.io.lammpsrun import read_lammps_dump

from subprocess import run
from wfl.configset import ConfigSet, OutputSpec


def lammps_run(input_structure_file, lammps_run_folder, lammps_binary, potential, temperature=300,
               total_steps=1000, timestep=0.001, dump_steps=50, verbose=False, **kwargs):
    """This function read in a configuration, runs a LAMMPS NVT simulation and transfers the simulation data back in
    a wfl compatible format. The script requires a lammps (mpi) binary. This is mainly a simple example of how to run
    LAMMPS automated within workflow and is supposed to be adjusted for the users need.


    Parameters
    ----------
    input_structure_file: str
        path to xyz or trajectorie file that can contain multiple fixed atoms
        (ase constraints)
    lammps_run_folder: str
        path to folder where the LAMMPS files will be is stored.
    lammps_binary: str
        path to mpi LAMMPS binary.
    potential: str
        two possibilities are currently supported:
            1. full LAMMPS pair_style command
            2. path to GAP file; path must include GAP in name i.e. GAP_14.xml; pair_style command will automatically
               be created and adjusted depending on the input_structure_file. (ML-QUIP plugin needed for LAMMPS)
    temperature: float or in
        temperature of the simulation
    total_steps: int, default = 1000
        amount of MD steps
    timestep: float, default = 0.001
        time in picoseconds
    dump_steps: int, default = 50
        every N steps frame will be added to xyz file
    verbose: bool
        prints extra information if True

    **kwargs:
        max_cores: int,
            maximum number of cores used, takes all cores available if not specified
        dump_filename: str,
            prefix for the xyz_file


    Returns
    -------
    None
    """

    # Setting up file Structure ######################################################################################
    if not lammps_run_folder.endswith('/'):
        lammps_run_folder += '/'

    lammps_run_file = lammps_run_folder + 'lammps.in'
    lammps_data_file = lammps_run_folder + 'lammps.data'
    lammps_out_file = lammps_run_folder + 'lammps.out'
    log_file = lammps_run_folder + 'lammps.log'
    dump_filename = lammps_run_folder + kwargs.pop('dump_filename', 'dump')
    initial_dump = lammps_run_folder + 'initial_cell.dump'

    # Extracting information from configuration ######################################################################
    atoms = read(input_structure_file, '-1')
    with open(lammps_data_file, 'w') as f:
        write_lammps_data(f, atoms, atom_style='full')

    # Order important for ase to lammps and lammps to ase
    chemical_sorts = list(np.unique(atoms.get_chemical_symbols()))
    chemical_sorts.sort()
    atom_sorts = [chemical_symbols.index(i) for i in chemical_sorts]

    fix_list = kwargs.pop('fix_commands', [])
    group_list = kwargs.pop('group_commands', [])

    # Adjusting for fixed and moving atoms ###########################################################################
    try:
        fixed_atoms_id = atoms.constraints[-1].index
        tmp_string = ''
        for val in fixed_atoms_id:
            tmp_string += f' {val + 1}'
        group_list.append('group fixed_layer id' + tmp_string)
        tmp_string = ''
        for val in range(atoms.get_global_number_of_atoms()):
            if val in fixed_atoms_id:
                continue
            else:
                tmp_string += f' {val + 1}'
        group_list.append('group moving_layer id' + tmp_string)
        group_list.append('compute moving_temp moving_layer temp')
        fix_list.append('fix fixedlayers fixed_layer setforce 0. 0. 0.')
        fix_list.append('fix extra all print ' + str(
            dump_steps) + ' """{"timestep": $(step), "pe":$(c_thermo_pe), "temp": $(c_moving_temp)}""" title "" file ' + dump_filename + '.json screen no')
        fix_list.append(
            f'fix thermo_fix all nvt temp {temperature} {temperature} 0.01 \nfix_modify thermo temp moving_temp')
        fix_list.append(f'velocity moving_layer create {temperature} {random.randint(1, 10000)}')

    except:
        fix_list.append('fix extra all print ' + str(
            dump_steps) + ' """{"timestep": $(step), "pe":$(c_thermo_pe), "temp": $(c_thermo_temp)}""" title "" file ' + dump_filename + '.json screen no')
        fix_list.append('fix momentum all momentum 100 linear 1 1 1 angular')
        fix_list.append(f'fix thermo_fix all nvt temp {temperature} {temperature} 0.01')
        fix_list.append(f'velocity all create {temperature} {random.randint(1, 10000)}')

    # Defining xzy Dumps #############################################################################################
    fix_list.append(f'write_dump all custom {initial_dump} id type x y z modify sort id')
    fix_list.append(
        f'dump all_xyz all xyz {dump_steps} {dump_filename}_tmp.xyz \ndump_modify all_xyz element {elements_to_string(atom_sorts)}')
    fix_list.append(f'timestep {timestep}')
    fix_list.append(f'run {total_steps}')

    group_string = ' \n'.join(group_list)
    fix_string = ' \n'.join(fix_list)

    # Set up Potential and write LAMMPS file #########################################################################
    with open(os.path.join(os.path.dirname(__file__), 'lammps_simple.txt'), 'r') as f:
        text_string = f.read()

    if 'pair_style' in potential:
        text_string = re.sub(r'potential_sign', potential, text_string)
    elif 'GAP' in potential:
        text_string = re.sub(r'potential_sign',
                             f'pair_style quip \npair_coeff * * {potential} "{get_GAP_id(potential)}" {atom_sorts_to_string(atom_sorts)}',
                             text_string)
    else:
        sys.exit('ERROR: pair_style not defined and potential not yet supported !!!')

    text_string = re.sub(r'input_file_sign', lammps_data_file, text_string)
    text_string = re.sub(r'masses_sign', masses_to_string(atom_sorts), text_string)

    text_string = re.sub(r'group_sign', group_string, text_string)
    text_string = re.sub(r'fix_sign', fix_string, text_string)

    with open(lammps_run_file, 'w') as f:
        f.write(text_string)

    if verbose:
        print(text_string +'\n\n')
        print('Simulation started at ', dt.datetime.now())

    # running LAMMPS #################################################################################################
    n_cores = kwargs.pop('max_cores', os.cpu_count())
    run(f'mpirun -n {n_cores} {lammps_binary} -l {log_file} -i {lammps_run_file} > {lammps_out_file}',
        shell=True)

    write_full_xyz(initial_dump, dump_filename)

    if verbose:
        print('Simulation finished at', dt.datetime.now())


def write_full_xyz(initial_dump, dump_filename):
    cell = read_lammps_dump(infileobj=initial_dump, order=False, index=-1).get_cell()
    in_config = ConfigSet(items=dump_filename + '_tmp.xyz')
    out_config = OutputSpec(files=dump_filename + '.xyz')
    data = pd.read_json(dump_filename + '.json', lines=True)['pe'].to_list()
    for i, at in enumerate(in_config):
        at.set_cell(cell)
        at.set_pbc(True)
        at.info['config_type'] = 'LAMMPS_MD'
        at.info['calc_energy'] = data[i]
        out_config.store(at)
    out_config.close()
    os.remove(dump_filename + '_tmp.xyz')


def get_GAP_id(gap_file):
    with open(gap_file, 'r') as f:
        string = f.read()
    gap_id = re.search(r'(\bIP GAP label=GAP)([0-9_]+)', string).group()
    return gap_id


def atom_sorts_to_string(atom_sorts):
    tmp_string = ''
    for val in atom_sorts:
        tmp_string += f' {val}'
    return tmp_string


def elements_to_string(atom_sorts):
    tmp_string = ''
    for i, val in enumerate(atom_sorts):
        tmp_string += f'{chemical_symbols[val]} '
    return tmp_string


def masses_to_string(atom_sorts):
    tmp_string = ''
    for i, val in enumerate(atom_sorts):
        tmp_string += f'mass {i + 1} {atomic_masses[val]} \n'
    return tmp_string
