import re, os, random

import numpy as np

from ase import Atoms
from ase.data import atomic_masses, chemical_symbols
from ase.io import read
from ase.io.lammpsdata import write_lammps_data
from ase.io.lammpsrun import read_lammps_dump
from ase.calculators.lammps import Prism, convert
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

    full_elements = [1, 8, 29]

    if not os.path.exists(run_id):
        os.mkdir(run_id)


    if not lammps_run_folder.endswith('/'):
        lammps_run_folder += '/'
        
        
    for (root, direct, files) in os.walk(lammps_run_folder):
        for file in files:
            if file.endswith('.traj'):
                input_structure_file = lammps_run_folder + file
            elif file.endswith('.xyz'):
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
    remaining_sorts = [i for i in full_elements if i not in atom_sorts]

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
            'fix extra all print 10 """{"timestep": $(step), "pe":$(c_thermo_pe), "n_pe": $(c_custom_pe), "temp": $(c_thermo_temp)}""" title "" file ' + dump_filename + '.${t}.json screen no')
        fix_list.append('fix momentum all momentum 100 linear 1 1 1 angular')
        fix_atoms_parameter = False


    group_string = ' \n'.join(group_list)
    fix_string = ' \n'.join(fix_list)

    with open(lammps_data_file, 'w') as f:
        write_lammps_data_custom(f, atoms, atom_style='full')

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
    for i, val in enumerate(atom_sorts):
        if val == 1:
            tmp_string += f'mass {i + 1} {atomic_masses[val]*2} \n'
        else:
            tmp_string += f'mass {i + 1} {atomic_masses[val]} \n'
    for i, val in enumerate(remaining_sorts):
        if val == 1:
            tmp_string += f'mass {i + 1} {atomic_masses[val]*2} \n'
        else:
            tmp_string += f'mass {i + 1} {atomic_masses[val]} \n'
    text_string = re.sub(r'masses_sign \n', tmp_string, text_string)

    tmp_string = ''
    for i, val in enumerate(atom_sorts):
        tmp_string += f'{chemical_symbols[val]} '
    for i, val in enumerate(remaining_sorts):
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
    
        
        


def write_lammps_data_custom(fd, atoms, specorder=None, force_skew=False,
                  prismobj=None, velocities=False, units="metal",
                  atom_style='atomic'):


    if isinstance(atoms, list):
        if len(atoms) > 1:
            raise ValueError(
                "Can only write one configuration to a lammps data file!"
            )
        atoms = atoms[0]

    if hasattr(fd, "name"):
        fd.write("{0} (written by ASE) \n\n".format(fd.name))
    else:
        fd.write("(written by ASE) \n\n")

    symbols = atoms.get_chemical_symbols()
    n_atoms = len(symbols)
    fd.write("{0} \t atoms \n".format(n_atoms))

    if specorder is None:
        # This way it is assured that LAMMPS atom types are always
        # assigned predictably according to the alphabetic order
        species = sorted(set(symbols))
    else:
        # To index elements in the LAMMPS data file
        # (indices must correspond to order in the potential file)
        species = specorder
    n_atom_types = 3
    fd.write("{0}  atom types\n".format(n_atom_types))

    if prismobj is None:
        p = Prism(atoms.get_cell())
    else:
        p = prismobj

    # Get cell parameters and convert from ASE units to LAMMPS units
    xhi, yhi, zhi, xy, xz, yz = convert(p.get_lammps_prism(), "distance",
                                        "ASE", units)

    fd.write("0.0 {0:23.17g}  xlo xhi\n".format(xhi))
    fd.write("0.0 {0:23.17g}  ylo yhi\n".format(yhi))
    fd.write("0.0 {0:23.17g}  zlo zhi\n".format(zhi))

    if force_skew or p.is_skewed():
        fd.write(
            "{0:23.17g} {1:23.17g} {2:23.17g}  xy xz yz\n".format(
                xy, xz, yz
            )
        )
    fd.write("\n\n")

    # Write (unwrapped) atomic positions.  If wrapping of atoms back into the
    # cell along periodic directions is desired, this should be done manually
    # on the Atoms object itself beforehand.
    fd.write("Atoms \n\n")
    pos = p.vector_to_lammps(atoms.get_positions(), wrap=False)

    if atom_style == 'atomic':
        for i, r in enumerate(pos):
            # Convert position from ASE units to LAMMPS units
            r = convert(r, "distance", "ASE", units)
            s = species.index(symbols[i]) + 1
            fd.write(
                "{0:>6} {1:>3} {2:23.17g} {3:23.17g} {4:23.17g}\n".format(
                    *(i + 1, s) + tuple(r)
                )
            )
    elif atom_style == 'charge':
        charges = atoms.get_initial_charges()
        for i, (q, r) in enumerate(zip(charges, pos)):
            # Convert position and charge from ASE units to LAMMPS units
            r = convert(r, "distance", "ASE", units)
            q = convert(q, "charge", "ASE", units)
            s = species.index(symbols[i]) + 1
            fd.write("{0:>6} {1:>3} {2:>5} {3:23.17g} {4:23.17g} {5:23.17g}\n"
                     .format(*(i + 1, s, q) + tuple(r)))
    elif atom_style == 'full':
        charges = atoms.get_initial_charges()
        # The label 'mol-id' has apparenlty been introduced in read earlier,
        # but so far not implemented here. Wouldn't a 'underscored' label
        # be better, i.e. 'mol_id' or 'molecule_id'?
        if atoms.has('mol-id'):
            molecules = atoms.get_array('mol-id')
            if not np.issubdtype(molecules.dtype, np.integer):
                raise TypeError((
                    "If 'atoms' object has 'mol-id' array, then"
                    " mol-id dtype must be subtype of np.integer, and"
                    " not {:s}.").format(str(molecules.dtype)))
            if (len(molecules) != len(atoms)) or (molecules.ndim != 1):
                raise TypeError((
                    "If 'atoms' object has 'mol-id' array, then"
                    " each atom must have exactly one mol-id."))
        else:
            # Assigning each atom to a distinct molecule id would seem
            # preferableabove assigning all atoms to a single molecule id per
            # default, as done within ase <= v 3.19.1. I.e.,
            # molecules = np.arange(start=1, stop=len(atoms)+1, step=1, dtype=int)
            # However, according to LAMMPS default behavior,
            molecules = np.zeros(len(atoms), dtype=int)
            # which is what happens if one creates new atoms within LAMMPS
            # without explicitly taking care of the molecule id.
            # Quote from docs at https://lammps.sandia.gov/doc/read_data.html:
            #    The molecule ID is a 2nd identifier attached to an atom.
            #    Normally, it is a number from 1 to N, identifying which
            #    molecule the atom belongs to. It can be 0 if it is a
            #    non-bonded atom or if you don't care to keep track of molecule
            #    assignments.

        for i, (m, q, r) in enumerate(zip(molecules, charges, pos)):
            # Convert position and charge from ASE units to LAMMPS units
            r = convert(r, "distance", "ASE", units)
            q = convert(q, "charge", "ASE", units)
            s = species.index(symbols[i]) + 1
            fd.write("{0:>6} {1:>3} {2:>3} {3:>5} {4:23.17g} {5:23.17g} "
                     "{6:23.17g}\n".format(*(i + 1, m, s, q) + tuple(r)))
    else:
        raise NotImplementedError

    if velocities and atoms.get_velocities() is not None:
        fd.write("\n\nVelocities \n\n")
        vel = p.vector_to_lammps(atoms.get_velocities())
        for i, v in enumerate(vel):
            # Convert velocity from ASE units to LAMMPS units
            v = convert(v, "velocity", "ASE", units)
            fd.write(
                "{0:>6} {1:23.17g} {2:23.17g} {3:23.17g}\n".format(
                    *(i + 1,) + tuple(v)
                )
            )

    fd.flush()