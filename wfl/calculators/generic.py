import warnings

from ase import Atoms
from ase.calculators.calculator import all_changes

from wfl.autoparallelize import autoparallelize, autoparallelize_docstring
from wfl.utils.misc import atoms_to_list
from wfl.utils.parallel import construct_calculator_picklesafe
from .utils import save_results


def _run_autopara_wrappable(atoms, calculator, properties=None, output_prefix='_auto_', verbose=False, raise_calc_exceptions=False):
    """evaluates configs using an arbitrary calculator and store results in info/arrays entries
    or `SinglePointCalculator`.

    Defaults to wfl_num_inputs_per_python_subprocess=10, to avoid recreating the calculator for
    each configuration, unless calculator class defines a wfl_generic_default_autopara_info
    attribute in which case that value is used for the default.

    If `Atoms.info` contains 'WFL\_CALCULATOR\_INITIALIZER', 'WFL\_CALCULATOR\_ARGS' or
    'WFL\_CALCULATOR\_KWARGS', an individual calculator will be created for that `Atoms` object.
    The `initializer` and `*args` will be _overridden_ by the corresponding `Atoms.info` entries, but
    the `**kwargs` will be _modified_ (`dict.update`) by the `Atoms.info` entry.

    Parameters
    ----------
    atoms: ase.atoms.Atoms / list(Atoms)
        input configuration(s)
    calculator: Calculator / (initializer (callable), args (list), kwargs (dict))
        ASE calculator or routine to call to create calculator. If 'WFL\_CALCULATOR\_ARGS'
        `...\_INITIALIZER`, or `...\_KWARGS` are present in any `Atoms.info` dicts, calculator
        _must_ be a 3-tuple so that those `initializer`, `*args` or `**kwargs` can be used to
        override defaults.
    properties: list(str), default ['energy', 'forces', stress']
        Properties to request from calculator. If any are not present after calculation (e.g.
        stress for nonperiodic configurations), a warning will be printed.
    output_prefix: str, default _auto_
        String to prefix info/arrays key names where results will be stored.
        '_auto_' for automatically determining name of calculator constructor, and
        None for SinglePointCalculator instead of info/arrays.
    verbose : bool
        verbose output
    """

    if properties is None:
        properties = ['energy', 'forces', 'stress']
    try:
        calculator_default = construct_calculator_picklesafe(calculator)
        calculator_failure_message = None
    except Exception as exc:
        # if calculator constructor failed, it may still be fine if every atoms object has
        # enough info to construct its own calculator, but we won't know until later
        calculator_failure_message = str(exc)
        calculator_default = None

    if output_prefix == '_auto_':
        output_prefix = calculator.__class__.__name__ + '_'

    at_out = []
    for at in atoms_to_list(atoms):
        calculator_use = calculator_default
        if ("WFL_CALCULATOR_INITIALIZER" in at.info or
            "WFL_CALCULATOR_ARGS" in at.info or
            "WFL_CALCULATOR_KWARGS" in at.info):
            # create per-config Calculator
            try:
                initializer_use = at.info.get("WFL_CALCULATOR_INITIALIZER", calculator[0])
                args_use = at.info.get("WFL_CALCULATOR_ARGS", calculator[1])
                kwargs_use = calculator[2].copy()
                kwargs_use.update(at.info.get("WFL_CALCULATOR_KWARGS", {}))
                calculator_use = construct_calculator_picklesafe((initializer_use, args_use, kwargs_use))
            except Exception as exc:
                raise TypeError("calculators.generic.calculate got WFL_CALCULATOR_INITIALIZER, _ARGS, or _KWARGS "
                                f"but constructor failed, most likely because calculator wasn't a tuple (TypeError) "
                                "or original tuple had invalid element that wasn't overridden by `Atoms.info` entry. "
                                f"Constructor exception was '{exc}'")

        if calculator_use is None:
            raise ValueError(f"Failed to construct calculator, original attempt's exception was '{calculator_failure_message}'")
        at.calc = calculator_use

        calculation_succeeded = False
        try:
            # explicitly pass system_changes=all_changes because some calculators, e.g. ace.ACECalculator,
            # don't have that as default
            at.calc.calculate(at, properties=properties, system_changes=all_changes)
            calculation_succeeded = True
            if f'{output_prefix}calculation_failed' in at.info:
                del at.info[f'{output_prefix}calculation_failed']
        except Exception as exc:
            if raise_calc_exceptions:
                raise exc
            import sys
            # pytest seems to hide these warnings for some reason
            if "pytest" in sys.modules:
                print(f'WARNING: calculation failed with exception {exc}')
            warnings.warn(f'calculation failed with exception {exc}')
            at.info[f'{output_prefix}calculation_failed'] = True

        # clean up invalid properties, will be fixed in quip Potential soon?
        if hasattr(at.calc, "results") and 'virial' in at.calc.results:
            del at.calc.results['virial']

        if calculation_succeeded:
            save_results(at, properties, output_prefix)
        else:
            # avoid maintaining the reference to the calculator
            at.calc = None

        at_out.append(at)

    if isinstance(atoms, Atoms):
        return at_out[0]
    else:
        return at_out


def calculate(*args, **kwargs):
    calculator = kwargs.get("calculator")
    if calculator is None:
        calculator = args[2]

    default_autopara_info = getattr(calculator, "wfl_generic_default_autopara_info", {"num_inputs_per_python_subprocess": 10})

    return autoparallelize(_run_autopara_wrappable, *args, default_autopara_info=default_autopara_info, **kwargs)
autoparallelize_docstring(calculate, _run_autopara_wrappable, "Atoms")
