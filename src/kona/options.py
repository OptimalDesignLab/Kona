def make_optn_key_str(keys):
    return "".join(["['%s']"%k for k in keys])

def get_opt(optns, default, *keys):
    '''
    utility function to make it easier to work with
    nested options dictionaries

    Parameters
    ------------------
    optns: nested dict
    default: value to return if no option found

    *keys: string keys for the nested dictionary
    '''
    keys = list(keys)

    k = keys.pop(0)
    val = optns.get(k, default)
    if isinstance(val, dict) and bool(val) and bool(keys):
        return get_opt(val, default, *keys)
    return val

class BadKonaOption(Exception):

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

from kona.algorithms.util.merit import ObjectiveMerit
from kona.linalg.matrices.lbfgs import LimitedMemoryBFGS
from kona.algorithms.util.linesearch import StrongWolfe
from kona.linalg.solvers.krylov import STCG

defaults = {
    'max_iter'          : 100,
    'primal_tol'        : 1e-8,
    'constraint_tol'    : 1e-8,
    'info_file'         : 'kona_hist.dat',

    'merit_function' : {
        'type'          : ObjectiveMerit,
    },

    'quasi_newton' : {
        # common options
        'type'          : LimitedMemoryBFGS,
        'max_stored'    : 10,
        # LimitedMemorySR1 options
        'threshold'     : 1e-8
    },

    'line_search' : {
        # common options
        'type'          : StrongWolfe,
        'max_iter'      : 50,
        'decr_cond'     : 1e-4,
        'alpha_init'    : 1.0,
        # BackTracking options
        'alpha_min'     : 1e-4,
        'rdtn_factor'   : 0.3,
        # StrongWolfe options
        'curv_cond'     : 0.9,
        'alpha_max'     : 2.0,
    },

    'krylov_solver' : {
        'type'          : STCG,
        'max_iter'      : 5000,
        'rel_tol'       : 1e-8,
        'check_res'     : True,
        'out_file'      : 'kona_krylov.dat',
        # STCG options
        'radius'        : 1.0,
        'proj_cg'       : False,
    }
}
