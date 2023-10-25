import re, os, random, sys

import numpy as np

from ase.data import atomic_masses, chemical_symbols
from ase.io import read
from ase.io.lammpsdata import write_lammps_data
from ase.io.lammpsrun import read_lammps_dump
from wfl.configset import ConfigSet, OutputSpec

from subprocess import run
#from wfl.configset import ConfigSet, OutputSpec



def replica_exchange(lammps_run_folder, run_id, lammps_src_maschine,
                     temperatures, mace_file, total_steps=1000, exchange_steps=250, timestep=0.001,
                     dump_steps=50, restart_steps = 10000, **kwargs):
    """parallel tempering and replica exchange using LAMMPS. Only requirement: LAMMPS mpi with
    replica package enabled.

    Parameters
    ----------
    input_structure_file: str
        trajectorie file that can contain multiple fixed atoms
        (ase constraints)
    lammps_run_folder: str
        file where the LAMMPS script will be is stored
    lammps_data_file: str
        file where the LAMMPS structure will be is stored
    lammps_src_maschine: str
        path to mpi LAMMPS machine.
    temperatures: list of floats
        temperatures of the replicas
    mace_file: str
        path to GAP that is used for the MDs
    total_steps: int, default = 1000
        amount of MD steps
    exchange_steps: int, default = 250
        each N steps tempering tests for
    timestep: float, default = 0.001
        time in picosecond
    dump_steps: int, default = 50
        every N steps frame will be added to xyz file

    **kwargs:
        dump_filename: str,
            prefex of folder for the xyz_file
        thermostat: str,
            LAMMPS thermostat command with ID myfix, by default a
            nose-hover nvt thermostat



    Returns
    -------
    None
    """

    if not os.path.exists(run_id):
        os.mkdir(run_id)


    if not lammps_run_folder.endswith('/'):
        lammps_run_folder += '/'
        
        
    for (root, direct, files) in os.walk(lammps_run_folder):
        for file in files:
            if file.endswith('.traj'):
                input_structure_file = lammps_run_folder + file
        
    lammps_run_folder += f'{run_id}/'

    lammps_run_file = lammps_run_folder + 'lammps.temper'
    lammps_data_file = lammps_run_folder + 'lammps.data'
    lammps_out_file = lammps_run_folder + 'lammps.out'
    log_file = lammps_run_folder + 'lammps.log'
    screen_file = lammps_run_folder + 'screen'
    dump_filename = lammps_run_folder + kwargs.pop('dump_filename', 'dump')
    initial_dump = lammps_run_folder + 'initial_cell.dump'

    atoms = read(input_structure_file, '-1')
    chemical_sorts = list(np.unique(atoms.get_chemical_symbols()))
    chemical_sorts.sort()
    atom_sorts = [chemical_symbols.index(i) for i in chemical_sorts]

    fix_list = kwargs.pop('fix_commands', [])
    group_list = kwargs.pop('group_commands', [])

    h_id = [i for i,val in enumerate(atoms.get_chemical_symbols()) if val =='H']
    if len(h_id) !=0:
        tmp_string = ''
        for val in h_id:
            tmp_string += f' {val + 1}'
        group_list.append('group hydrogen id' + tmp_string)
        fix_list.append('fix hzwall_lo hydrogen wall/harmonic zlo EDGE 2.0 0.0 4.5')
    fix_list.append('fix hzwall_hi all wall/harmonic zhi EDGE 0.5 0.0 10.0')
    


    try:
        fixed_atoms_id = atoms.constraints[-1].index
        fix_atoms_parameter = True
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
#        group_list.append('compute vec_pe all pe/atom')
#        group_list.append('variable norm_pe atom 3*c_vec_pe')
#        group_list.append('compute custom_pe all reduce ave v_norm_pe')

        fix_list.append('fix fixedlayers fixed_layer setforce 0. 0. 0.')
        fix_list.append(
                'fix extra all print 10 """{"timestep": $(step), "pe":$(c_thermo_pe), "n_pe": $(c_custom_pe), "temp": $(c_moving_temp)}""" title "" file ' + dump_filename + '.${t}.json screen no')

    except:
        fix_list.append(
            'fix extra all print 10 """{"timestep": $(step), "pe":$(c_thermo_pe), "temp": $(c_thermo_temp)}""" title "" file ' + dump_filename + '.${t}.json screen no')
        fix_list.append('fix momentum all momentum 100 linear 1 1 1 angular')
        fix_atoms_parameter = False


    group_string = ' \n'.join(group_list)
    fix_string = ' \n'.join(fix_list)

    with open(lammps_data_file, 'w') as f:
        write_lammps_data(f, atoms, atom_style='full')

    template = os.path.join(os.path.dirname(os.path.realpath(__file__)),'lammps_remote.txt')
    print(template)
    with open(template, 'r') as f:
        text_string = f.read()

    text_string = re.sub(r'initial_cell_sign', initial_dump, text_string)
    text_string = re.sub(r'input_file_sign', lammps_data_file, text_string)
    text_string = re.sub(r'temperatures_sign', ' '.join(map(str, temperatures)), text_string)
    text_string = re.sub(r'timestep_sign', str(timestep), text_string)
    text_string = re.sub(r'total_steps_sign', str(total_steps), text_string)
    text_string = re.sub(r'exchange_steps_sign', str(exchange_steps), text_string)
    text_string = re.sub(r'dump_steps_sign', str(dump_steps), text_string)
    text_string = re.sub(r'dump_name_sign', dump_filename + '_tmp', text_string)
    text_string = re.sub(r'mace_file_sign', mace_file, text_string)
    text_string = re.sub(r'seed_sign', f'{random.randint(1, 10000)} {random.randint(1, 10000)}', text_string)
    
    if fix_atoms_parameter:
        text_string = re.sub(r'thermostat_sign', kwargs.pop('thermostat',
                                                            'fix myfix all nvt temp $t $t 0.01 \nfix_modify myfix temp moving_temp'),
                             text_string)
        text_string = re.sub(r'velocity_sign', f'velocity moving_layer create $t {random.randint(1, 10000)}',
                             text_string)
    else:
        text_string = re.sub(r'thermostat_sign', kwargs.pop('thermostat',
                                                            'fix myfix all nvt temp $t $t 0.01'),
                             text_string)
        text_string = re.sub(r'velocity_sign', f'velocity all create $t {random.randint(1, 10000)}',
                             text_string)
    

    text_string = re.sub(r'group_sign', group_string, text_string)
    text_string = re.sub(r'fix_sign', fix_string, text_string)
    text_string = re.sub(r'restart_sign', f'restart {restart_steps} ' + lammps_run_folder +'lammps${t}.restart', text_string)


    tmp_string = ''
    for val in atom_sorts:
        tmp_string += f'{val} '
    text_string = re.sub(r'atom_sort_sign', tmp_string, text_string)

    tmp_string = ''
    for i, val in enumerate(atom_sorts):
        if val == 1:
            tmp_string += f'mass {i + 1} {atomic_masses[val]*2} \n'
        else:
            tmp_string += f'mass {i + 1} {atomic_masses[val]} \n'
    text_string = re.sub(r'masses_sign \n', tmp_string, text_string)

    tmp_string = ''
    for i, val in enumerate(atom_sorts):
        tmp_string += f'{chemical_symbols[val]} '
    text_string = re.sub(r'elements_sign', tmp_string, text_string)
    
    with open(lammps_run_file, 'w') as f:
        f.write(text_string)
    
    used_cores = len(temperatures)

    partition_string = ''
    for val in temperatures:
        partition_string += f'1 '
   
    print(f'srun -n {used_cores} {lammps_src_maschine} -partition {partition_string}-l {log_file} -sc {screen_file} -k on g 1 -sf kk -i {lammps_run_file} > {lammps_out_file}', flush=True)
    run(f'srun -n {used_cores} {lammps_src_maschine} -partition {partition_string}-l {log_file} -sc {screen_file} -k on g 1 -sf kk -i {lammps_run_file} > {lammps_out_file}', shell=True)


    cell = read_lammps_dump(infileobj=initial_dump, order=False, index=-1).get_cell()

    for temp in temperatures:
        in_config = ConfigSet(items=dump_filename + f'_tmp.{temp}.xyz')
        out_config = OutputSpec(files=dump_filename + f'.{temp}.xyz')
        for at in in_config:
            at.set_cell(cell)
            at.set_pbc(True)
            at.info['config_type'] = 'parralell_tempering'
            out_config.store(at)
        out_config.close()
        os.remove(dump_filename + f'_tmp.{temp}.xyz')
    
        
        
