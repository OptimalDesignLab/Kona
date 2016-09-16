import numpy as np
from kona.linalg.vectors.composite import CompositePrimalVector
from kona.linalg.vectors.composite import CompositeDualVector

def current_solution(num_iter, curr_design, curr_state=None, curr_adj=None,
                     curr_eq=None, curr_ineq=None, curr_slack=None):
    """
    Notify the solver of the current solution point.

    Parameters
    ----------
    num_iter : int
        Current iteration of the optimization.
    curr_design : DesignVector
        Current design variables.
    curr_state : StateVector, optional
        Current state variables.
    curr_adj : StateVector, optional
        Current adjoint variables.
    curr_eq : DualVectorEQ, optional
        Current Lagrange multipliers for equality constraints.
    curr_ineq : DualVectorINEQ, optional
        Current Lagrange multipliers for inequality constraints.
    curr_slack : DualVectorINEQ, optional
        Current slack variables.
    """

    solver = curr_design._memory.solver
    curr_design = curr_design.base.data

    if curr_state is not None:
        curr_state = curr_state.base

    if curr_adj is not None:
        curr_adj = curr_adj.base

    if curr_eq is not None:
        curr_eq = curr_eq.base.data

    if curr_ineq is not None:
        curr_ineq = curr_ineq.base.data

    if curr_slack is not None:
        curr_slack = curr_slack.base.data

    return solver.current_solution(
        num_iter, curr_design, curr_state, curr_adj,
        curr_eq, curr_ineq, curr_slack)

def objective_value(at_design, at_state):
    """
    Evaluate the objective value the given Primal and State point.

    Parameters
    ----------
    at_design : DesignVector
        Current design point.
    at_state : StateVector
        Current state point.

    Returns
    -------
    float
        Objective function value.
    """

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

def lagrangian_value(at_kkt, at_state, dual_work, barrier=None):
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
    at_primal = at_kkt.primal
    at_dual = at_kkt.dual
    if isinstance(at_primal, CompositePrimalVector):
        if barrier is None:
            raise ValueError("lagrangian_value >> Missing arrier coefficient!")
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

    # evaluate objective
    lagrangian = objective_value(at_design, at_state)

    # evaluate constraints
    dual_work.equals_constraints(at_design, at_state)

    # add constraint terms
    if at_slack is not None:
        if at_dual_eq is not None:
            lagrangian += at_dual_eq.inner(dual_work.eq)
        dual_work.ineq.minus(at_slack)
        lagrangian += at_dual_ineq.inner(dual_work.ineq)
    else:
        lagrangian += at_dual_eq.inner(dual_work)

    # add log barrier
    lagrangian += 0.5*barrier*np.sum(np.log(at_slack.base.data))


def factor_linear_system(at_design, at_state):
    """
    Trigger the solver to factor and store the dR/dU matrix and its
    preconditioner, linearized at the given ``at_design`` and ``at_state``
    point.

    Parameters
    ----------
    at_design : DesignVector
    at_state : StateVector
    """
    solver = at_design._memory.solver

    if solver != at_state._memory.solver:
        raise MemoryError('objective_value() >> Primal and State ' +
                          'vectors are not on the same memory manager!')

    solver.factor_linear_system(at_design.base.data, at_state.base)