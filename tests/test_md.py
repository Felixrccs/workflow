from pytest import approx
import pytest

import numpy as np

from ase import Atoms
import ase.io
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.units import fs
from wfl.autoparallelize import autoparainfo

from wfl.generate import md
from wfl.configset import ConfigSet, OutputSpec
from wfl.generate.md.abort import AbortOnCollision


def select_every_10_steps_for_tests_during(at):
    return at.info.get("MD_step", 1) % 10 == 0

def select_every_10_steps_for_tests_after(traj):
    return [at for at in traj if at.info["MD_step"] % 10 == 0]

def check_validity_for_tests(at):
    if "5" in str(at.info["MD_step"]):
        return False
    return True

@pytest.fixture
def cu_slab():

    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    atoms *= (2, 2, 2)
    atoms.rattle(stdev=0.01, seed=159)

    atoms.info['config_type'] = 'cu_slab'
    atoms.info['buildcell_config_i'] = 'fake_buildecell_config_name'

    return atoms


def test_NVE(cu_slab):
    calc = EMT()

    inputs = ConfigSet(cu_slab)
    outputs = OutputSpec()

    atoms_traj = md.md(inputs, outputs, calculator=calc, steps=300, dt=1.0,
                           temperature = 500.0)

    atoms_traj = list(atoms_traj)
    atoms_final = atoms_traj[-1]

    assert len(atoms_traj) == 301


def test_NVT_const_T(cu_slab):

    calc = EMT()

    inputs = ConfigSet(cu_slab)
    outputs = OutputSpec()

    atoms_traj = md.md(inputs, outputs, calculator=calc, steps=300, dt=1.0,
                           temperature = 500.0, temperature_tau=30.0)

    atoms_traj = list(atoms_traj)
    atoms_final = atoms_traj[-1]

    assert len(atoms_traj) == 301
    assert all([at.info['MD_temperature_K'] == 500.0 for at in atoms_traj])


def test_NVT_Langevin_const_T(cu_slab):

    calc = EMT()

    inputs = ConfigSet(cu_slab)
    outputs = OutputSpec()

    atoms_traj = md.md(inputs, outputs, calculator=calc, integrator="Langevin", steps=300, dt=1.0,
                           temperature = 500.0, temperature_tau=100/fs)

    atoms_traj = list(atoms_traj)
    atoms_final = atoms_traj[-1]

    assert len(atoms_traj) == 301
    assert all([at.info['MD_temperature_K'] == 500.0 for at in atoms_traj])


def test_NVT_const_T_mult_configs_distinct_seeds(cu_slab):

    calc = EMT()

    inputs = ConfigSet([cu_slab.copy() for _ in range(4)])
    outputs = OutputSpec()

    atoms_traj = md.md(inputs, outputs, calculator=calc, steps=300, dt=1.0,
                           temperature = 500.0, temperature_tau=30.0, autopara_rng_seed=23875)

    last_configs = [list(group)[-1] for group in atoms_traj.groups()]
    last_vs = [np.linalg.norm(at.get_velocities()) for at in last_configs]
    print("BOB last_vs", last_vs)
    assert all([v != last_vs[0] for v in last_vs[1:]])


def test_NVT_simple_ramp(cu_slab):

    calc = EMT()

    inputs = ConfigSet(cu_slab)
    outputs = OutputSpec()

    atoms_traj = md.md(inputs, outputs, calculator=calc, steps=300, dt=1.0,
                           temperature = (500.0, 100.0), temperature_tau=30.0)

    atoms_traj = list(atoms_traj)
    atoms_final = atoms_traj[-1]

    assert len(atoms_traj) == 301
    Ts = []
    for T_i, T in enumerate(np.linspace(500.0, 100.0, 10)):
        Ts.extend([T] * 30)
        if T_i == 0:
            Ts.append(T)
    assert all(np.isclose(Ts, [at.info['MD_temperature_K'] for at in atoms_traj]))


def test_NVT_complex_ramp(cu_slab):

    calc = EMT()

    inputs = ConfigSet(cu_slab)
    outputs = OutputSpec()

    atoms_traj = md.md(inputs, outputs, calculator=calc, steps=300, dt=1.0,
                           temperature = [{'T_i': 100.0, 'T_f': 500.0, 'traj_frac': 0.5},
                                          {'T_i': 500.0, 'T_f': 500.0, 'traj_frac': 0.25},
                                          {'T_i': 500.0, 'T_f': 300.0, 'traj_frac': 0.25}],
                           temperature_tau=30.0)

    atoms_traj = list(atoms_traj)
    atoms_final = atoms_traj[-1]

    assert len(atoms_traj) == 306
    Ts = []
    for T_i, T in enumerate(np.linspace(100.0, 500.0, 10)):
        Ts.extend([T] * 15)
        if T_i == 0:
            Ts.append(T)
    Ts.extend([500.0] * 75)
    for T_i, T in enumerate(np.linspace(500.0, 300.0, 10)):
        Ts.extend([T] * 8)

    # for at_i, at in enumerate(atoms_traj):
        # print(at_i, at.info['MD_time_fs'], 'MD', at.info['MD_temperature_K'], 'test', Ts[at_i])

    assert all(np.isclose(Ts, [at.info['MD_temperature_K'] for at in atoms_traj]))


def test_subselector_function_after(cu_slab):

    calc = EMT()

    inputs = ConfigSet(cu_slab)
    outputs = OutputSpec()

    atoms_traj = md.md(inputs, outputs, calculator=calc, steps=300, dt=1.0,
                           temperature = 500.0, traj_select_after_func=select_every_10_steps_for_tests_after)

    atoms_traj = list(atoms_traj)
    assert len(atoms_traj) == 31


def test_subselector_function_during(cu_slab):

    calc = EMT()

    for steps in [300, 301]:
        inputs = ConfigSet(cu_slab)
        outputs = OutputSpec()

        atoms_traj = md.md(inputs, outputs, calculator=calc, steps=steps, dt=1.0,
                               temperature = 500.0, traj_select_during_func=select_every_10_steps_for_tests_during)

        atoms_traj = list(atoms_traj)
        assert len(atoms_traj) == 31


def test_md_abort_function(cu_slab):

    calc = EMT()

    inputs = ConfigSet(cu_slab)
    outputs = OutputSpec()

    md_stopper = AbortOnCollision(collision_radius=2.25)
    autopara_info = autoparainfo.AutoparaInfo(skip_failed=False)

    # why doesn't this throw an raise a RuntimeError even if md failed and `skip_failed` is False?
    atoms_traj = md.md(inputs, outputs, calculator=calc, steps=500, dt=10.0,
                           temperature = 2000.0, abort_check=md_stopper, autopara_info=autopara_info) 

    assert len(list(atoms_traj)) < 501
