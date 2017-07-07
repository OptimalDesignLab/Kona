import numpy as np
from copy import deepcopy
from kona.linalg.vectors.common import DesignVector, StateVector
from kona.linalg.vectors.common import DualVectorEQ, DualVectorINEQ
from kona.linalg.vectors.composite import ReducedKKTVector
from kona.linalg.vectors.composite import CompositePrimalVector
from kona.linalg.vectors.composite import CompositeDualVector

def current_solution(num_iter, curr_primal, curr_state=None, curr_adj=None,
                     curr_dual=None):
    """
    Notify the solver of the current solution point.

    Parameters
    ----------
    num_iter : int
        Current iteration of the optimization.
    curr_primal : DesignVector or CompositePrimalVector
        Current design variables.
    curr_state : StateVector, optional
        Current state variables.
    curr_adj : StateVector, optional
        Current adjoint variables.
    curr_dual : DualVectorEQ, DualVectorINEQ or CompositeDualVector
    """
    if isinstance(curr_primal, CompositePrimalVector):
        curr_design = curr_primal.design
        out_slack = deepcopy(curr_primal.slack.base.data)
        if isinstance(curr_dual, DualVectorINEQ):
            out_eq = None
            out_ineq = deepcopy(curr_dual.base.data)
        elif isinstance(curr_dual, CompositeDualVector):
            out_eq = deepcopy(curr_dual.eq.base.data)
            out_ineq = deepcopy(curr_dual.ineq.base.data)
        else:
            raise TypeError("Invalid dual vector type: " +
                            "must be DualVectorINEQ or CompositeDualVector!")

    elif isinstance(curr_primal, DesignVector):
        curr_design = curr_primal
        out_slack = None
        if isinstance(curr_dual, DualVectorINEQ):
            out_eq = None
            out_ineq = deepcopy(curr_dual.base.data)
        elif isinstance(curr_dual, DualVectorEQ):
            out_eq = deepcopy(curr_dual.base.data)
            out_ineq = None
        elif isinstance(curr_dual, CompositeDualVector):
            out_eq = deepcopy(curr_dual.eq.base.data)
            out_ineq = deepcopy(curr_dual.ineq.base.data)
        elif curr_dual is None:
            out_eq = None
            out_ineq = None
        else:
            raise TypeError("Invalid dual vector type")

    else:
        raise TypeError("Invalid primal vector type: " +
                        "must be DesignVector or CompositePrimalVector!")

    solver = curr_design._memory.solver
    out_design = deepcopy(curr_design.base.data)

    if isinstance(curr_state, StateVector):
        out_state = curr_state.base
    elif curr_state is None:
        out_state = None
    else:
        raise TypeError("Invalid state vector type: must be StateVector!")

    if isinstance(curr_adj, StateVector):
        out_adj = curr_adj.base
    elif curr_adj is None:
        out_adj = None
    else:
        raise TypeError("Invalid adjoint vector type: must be StateVector!")

    return solver.current_solution(
        num_iter, out_design, out_state, out_adj, out_eq, out_ineq, out_slack)

def objective_value(at_primal, at_state):
    """
    Evaluate the objective value the given Primal and State point.

    Parameters
    ----------
    at_primal : DesignVector or CompositePrimalVector
        Current design point.
    at_state : StateVector
        Current state point.

    Returns
    -------
    float
        Objective function value.
    """
    if isinstance(at_primal, CompositePrimalVector):
        at_design = at_primal.design
    elif isinstance(at_primal, DesignVector):
        at_design = at_primal
    else:
        raise TypeError("Invalid primal vector type: " +
                        "must be DesignVector or CompositePrimalVector!")
    assert isinstance(at_state, StateVector), \
        "Invalid state vector type: must be StateVector!"

    solver = at_design._memory.solver

    result = solver.eval_obj(at_design.base.data, at_state.base)

    if isinstance(result, tuple):
        at_design._memory.cost += result[1]
        return result[0]
    elif isinstance(result, float):
        return result
    else:
        raise TypeError(
            'objective_value() >> solver.eval_obj() ' +
            'expected 2-tuple or float but was given %s'%type(result))

def lagrangian_value(at_kkt, at_state, barrier=None):
    """
    Evaluate the lagrangian at the given KKT and state point.

    Parameters
    ----------
    at_kkt : ReducedKKTVector
        Primal-dual point for evaluating the lagrangian.
    at_state : StateVector
        State point for evaluating the lagrangian
    barrier : float, optional
        Coefficient for logarithmic barrier term for slack non-negativity.

    Returns
    -------
    float
        Lagrangian value.
    """
    # do some vector aliasing
    assert isinstance(at_kkt, ReducedKKTVector), \
        "Invalid KKT vector type: must be ReducedKKTVector!"
    at_primal = at_kkt.primal
    at_dual = at_kkt.dual
    if isinstance(at_primal, CompositePrimalVector):
        assert barrier is not None, "Missing barrier coefficient!"
        at_design = at_kkt.primal.design
        at_slack = at_kkt.primal.slack
        if isinstance(at_dual, CompositeDualVector):
            at_dual_eq = at_kkt.dual.eq
            at_dual_ineq = at_kkt.dual.ineq
        else:
            at_dual_eq = None
            at_dual_ineq = at_dual
    else:
        at_design = at_kkt.primal
        at_slack = None
        at_dual_eq = at_dual
        at_dual_ineq = None

    # get solver handle
    solver = at_design._memory.solver

    # evaluate objective
    lagrangian = objective_value(at_design, at_state)

    # add constraint terms
    if at_slack is not None:
        if at_dual_eq is not None:
            eq_cnstr = solver.eval_eq_cnstr(
                at_design.base.data, at_state.base)
            lagrangian += np.dot(at_dual_eq.base.data, eq_cnstr)
        ineq_cnstr = solver.eval_ineq_cnstr(at_design.base.data, at_state.base)
        lagrangian += np.dot(at_dual_ineq.base.data, ineq_cnstr)
    else:
        eq_cnstr = solver.eval_eq_cnstr(
            at_design.base.data, at_state.base)
        lagrangian += np.dot(at_dual_eq.base.data, eq_cnstr)

    # add log barrier
    if at_slack is not None:
        lagrangian += 0.5 * barrier * np.sum(np.log(at_slack.base.data))

    return lagrangian

def factor_linear_system(at_primal, at_state):
    """
    Trigger the solver to factor and store the dR/dU matrix and its
    preconditioner, linearized at the given ``at_primal`` and ``at_state``
    point.

    Parameters
    ----------
    at_primal : DesignVector or CompositePrimalVector
    at_state : StateVector
    """
    if isinstance(at_primal, CompositePrimalVector):
        at_design = at_primal.design
    else:
        at_design = at_primal

    solver = at_design._memory.solver

    solver.factor_linear_system(at_design.base.data, at_state.base)
