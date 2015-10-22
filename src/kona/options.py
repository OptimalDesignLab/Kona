def make_optn_key_str(keys):
    return "".join(["['%s']"%k for k in keys])

def get_opt(optns, default, *keys):
    """
    Utility function to make it easier to work with nested options dictionaries.

    Parameters
    ----------
    optns : dict
        Nested dictionary.
    default : Unknown
        Value to return of the dictionary is empty.
    \*keys : string
        Keys from which value will be pulled

    Returns
    -------
    Unknown
        Dictionary value corresponding to given hierarchy of keys.
    """
    keys = list(keys)

    k = keys.pop(0)
    val = optns.get(k, default)
    if isinstance(val, dict) and bool(val) and bool(keys):
        return get_opt(val, default, *keys)
    return val

class BadKonaOption(Exception):
    """
    Special exception class for identifying bad Kona configuration options.

    Parameters
    ----------
    optns : dict
        Options dictionary containing the bad configuration.
    \*keys : string
        Hierarchy of dictionary keys identifying the bad configuration.
    """
    def __init__(self, optns, *keys):
        self.val  = get_opt(optns, None, *keys)
        self.keys = keys

    def __str__(self):
        optns_str = make_optn_key_str(self.keys)
        return "Invalid Kona option: optns%s = %s" % (optns_str, self.val)

# IMPORTANT
#############
# Default options go to the bottom of this file to remove circular import errors
#############
# IMPORTANT

import sys

from kona.algorithms.util.merit import ObjectiveMerit
from kona.linalg.matrices.hessian import LimitedMemoryBFGS
from kona.algorithms.util.linesearch import StrongWolfe
from kona.linalg.solvers.krylov import STCG

defaults = {
    'max_iter'          : 100,
    'primal_tol'        : 1e-8,
    'constraint_tol'    : 1e-8,
    'info_file'         : sys.stdout,
    'hist_file'         : 'kona_hist.dat',
    'matrix_explicit'   : False,

    'merit_function' : {
        'type'          : ObjectiveMerit,
    },

    'trust' : {
        'init_radius'   : 0.5,
        'max_radius'    : 0.5*(2**3),
        'min_radius'    : 0.5/(2**3),
    },

    'aug_lag' : {
        'mu_init' : 10.0,
        'mu_pow' : 1.0,
        'mu_max' : 1e5,
    },

    'reduced' : {
        'precond'       : None,
        'product_fac'   : 0.001,
        'lambda'        : 0.0,
        'scale'         : 0.0,
        'nu'            : 0.95,
        'dynamic_tol'   : False,
    },

    'quasi_newton' : {
        # common options
        'type'          : LimitedMemoryBFGS,
        'max_stored'    : 10,
        # LimitedMemorySR1 options
        'threshold'     : 1e-8
    },

    'globalization' : {
        # common options
        'type'          : StrongWolfe,
        'max_iter'      : 50,
        'decr_cond'     : 1e-4,
        'alpha_init'    : 1.0,
        # BackTracking options
        'alpha_min'     : 1e-4,
        'rdtn_factor'   : 0.5,
        # StrongWolfe options
        'curv_cond'     : 0.9,
        'alpha_max'     : 2.0,
    },

    'krylov' : {
        'out_file'      : 'kona_krylov.dat',
        'type'          : STCG,
        'max_iter'      : 10,
        'rel_tol'       : 0.05,
        'check_res'     : True,
        'proj_cg'       : False, # STCG
        'grad_scale'    : 1.0, # FLECS
        'feas_scale'    : 1.0, # FLECS
    },

    'verify' : {
        'primal_vec'    : True,
        'state_vec'     : True,
        'dual_vec'      : False,
        'gradients'     : True,
        'pde_jac'       : True,
        'cnstr_jac'     : False,
        'red_grad'      : True,
        'lin_solve'     : False,
        'out_file'      : sys.stdout,
    },
}
