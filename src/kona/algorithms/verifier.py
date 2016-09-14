import numpy, sys

from kona.options import get_opt
from kona.linalg import objective_value, factor_linear_system
from kona.linalg.solvers.util import calc_epsilon
from kona.linalg.matrices.common import dRdX, dRdU, dCdX, dCdU
from kona.linalg.vectors.common import PrimalVector, StateVector, DualVector

EPS = numpy.finfo(numpy.float32).eps

class Verifier(object):
    """
    This is a verification tool that performs finite-difference checks on the
    provided user solver to make sure that the required optimization tasks
    have been implemented correctly.

    Attributes
    ----------
    primal_factory : VectorFactory
    state_factory : VectorFactory
    dual_factory : VectorFactory
    optns : dict
    out_stream :

    Parameters
    ----------
    vector_factories : list of VectorFactory
    solver : UserSolver
    optns : dict
    """
    def __init__(self, vector_factories, solver, optns={}):
        # extract vector factories
        self.primal_factory = None
        self.state_factory = None
        self.dual_factory = None
        for factory in vector_factories:
            if factory is not None:
                if factory._vec_type is PrimalVector:
                    self.primal_factory = factory
                if factory._vec_type is StateVector:
                    self.state_factory = factory
                if factory._vec_type is DualVector:
                    self.dual_factory = factory

        # store solver handle
        self.solver = solver

        # assemble internal options dictionary
        self.optns = {
            'primal_vec' : get_opt(optns, True, 'primal_vec'),
            'state_vec'  : get_opt(optns, True, 'state_vec'),
            'dual_vec'   : get_opt(optns, False, 'dual_vec'),
            'gradients'  : get_opt(optns, True, 'gradients'),
            'pde_jac'    : get_opt(optns, True, 'pde_jac'),
            'cnstr_jac'  : get_opt(optns, False, 'cnstr_jac'),
            'red_grad'   : get_opt(optns, True, 'red_grad'),
            'lin_solve'  : get_opt(optns, True, 'lin_solve'),
        }
        self.out_stream = get_opt(optns, sys.stdout, 'out_file')
        self.factor_matrices = get_opt(optns, False, 'matrix_explicit')

        # request vectors
        num_primal = 0
        num_state = 0
        num_dual = 0
        if self.optns['primal_vec']:
            num_primal = max(num_primal, 3)
        if self.optns['state_vec']:
            num_state = max(num_state, 3)
        if self.optns['dual_vec']:
            num_dual = max(num_dual, 3)
        if self.optns['gradients']:
            num_primal = max(num_primal, 4)
            num_state = max(num_state, 4)
        if self.optns['pde_jac']:
            num_primal = max(num_primal, 6)
            num_state = max(num_state, 6)
        if self.optns['cnstr_jac']:
            num_primal = max(num_primal, 6)
            num_state = max(num_state, 6)
            num_dual = max(num_dual, 6)
        if self.optns['red_grad']:
            num_primal = max(num_primal, 4)
            num_state = max(num_state, 4)
        if self.optns['lin_solve']:
            num_primal = max(num_primal, 1)
            num_state = max(num_state, 5)
        self.primal_factory.request_num_vectors(num_primal)
        self.state_factory.request_num_vectors(num_state)
        if self.optns['dual_vec']:
            self.dual_factory.request_num_vectors(num_dual)

        # set a dictionary that will keep track of failures
        self.warnings_flagged = False
        self.exit_verify = False
        self.failures = {
            # BaseVector algebra operations
            'primal_vec' : {
                'equals_value'          : None,
                'equals_vector'         : None,
                'plus'                  : None,
                'times'                 : None,
                'equals_ax_p_by'        : None,
                'exp'                   : None,
                'log'                   : None,
                'pow'                   : None,
                'inner'                 : None,
            },
            'state_vec' : {
                'equals_value'          : None,
                'equals_vector'         : None,
                'plus'                  : None,
                'times'                 : None,
                'equals_ax_p_by'        : None,
                'exp'                   : None,
                'log'                   : None,
                'pow'                   : None,
                'inner'                 : None,
            },
            'dual_vec' : {
                'equals_value'          : None,
                'equals_vector'         : None,
                'plus'                  : None,
                'times'                 : None,
                'equals_ax_p_by'        : None,
                'exp'                   : None,
                'log'                   : None,
                'pow'                   : None,
                'inner'                 : None,
            },
            # UserSolver gradient operations
            'gradients' : {
                'eval_dFdX'             : None,
                'eval_dFdU'             : None,
            },
            # UserSolver PDE jacobian operations
            'pde_jac' : {
                'multiply_dRdX'         : None,
                'multiply_dRdU'         : None,
                'multiply_dRdX_T'       : None,
                'multiply_dRdU_T'       : None,
            },
            # UserSolver constraint jacobian operations
            'cnstr_jac' : {
                'multiply_dCdX'         : None,
                'multiply_dCdU'         : None,
                'multiply_dCdX_T'       : None,
                'multiply_dCdU_T'       : None,
            },
            # UserSolver forward and reverse linear solves
            'lin_solve' : {
                'solve_linear'          : None,
            },
            'red_grad' : {
                'solve_adjoint'         : None,
            },
        }

        # define an array if useful keys
        self.critical = ['primal_vec', 'state_vec', 'dual_vec']
        self.non_critical = \
            ['gradients', 'pde_jac', 'cnstr_jac', 'lin_solve', 'red_grad']
        self.all_tests = self.critical + self.non_critical

    def solve(self):
        # loop through all verification operations
        for op_name in self.all_tests:
            # if the operation is marked for verification
            if self.optns[op_name]:
                # reset failures to False
                for function in self.failures[op_name]:
                    self.failures[op_name][function] = False
                # run the verification to determine failures
                self.__getattribute__('_verify_' + op_name)()
                # if the verification produced a severe error, exit
                if self.exit_verify:
                    break
        # print verification report
        self._print_failure_report()

    def _print_failure_report(self):
        self.out_stream.write(
            '============================================================\n' +
            'Verification Report\n' +
            '------------------------------\n'
        )
        # failures in vector space operations are critical errors
        # other tests cannot work properly without these operations
        # therefore we will print these errors and cut the report short
        for op_name in self.critical:
            end_report = False
            for function in self.failures[op_name]:
                if self.failures[op_name][function]:
                    end_report = True
                    self.out_stream.write(
                        '%10s : %20s... '%(op_name[:-4], function) +
                        'CRITICAL FAILURE! Check for errors\n'
                    )
            if end_report:
                return
        # if we get here, that means vector space operations work fine
        # failures in other operations are printed as a full report
        for op_name in self.non_critical:
            for function in self.failures[op_name]:
                if self.failures[op_name][function] is None:
                    result = 'Untested'
                elif isinstance(self.failures[op_name][function], bool):
                    if self.failures[op_name][function]:
                        result = 'WARNING! Possible errors'
                    else:
                        result = 'Passed'
                self.out_stream.write(
                    ('%s'%function).ljust(20).replace(' ', '.') +
                    '...%s\n'%result
                )

    def _verify_primal_vec(self):
        if not self.optns['primal_vec']:
            return

        u_p = self.primal_factory.generate()
        v_p = self.primal_factory.generate()
        w_p = self.primal_factory.generate()

        u_p.equals(10.0)
        u_p.divide_by(u_p.norm2)
        one = u_p.norm2
        if abs(one) - 1 > EPS:
            self.failures['primal_vec']['equals_value'] = True
            self.failures['primal_vec']['times'] = True
            self.failures['primal_vec']['inner'] = True
            self.exit_verify = True

        u_p.equals(1.)
        v_p.equals(-1.)
        w_p.equals(u_p)
        w_p.plus(v_p)
        zero = w_p.norm2
        if abs(zero) > EPS:
            self.failures['primal_vec']['equals_value'] = True
            self.failures['primal_vec']['equals_vector'] = True
            self.failures['primal_vec']['plus'] = True
            self.failures['primal_vec']['inner'] = True
            self.exit_verify = True

        u_p.times(2.)
        v_p.divide_by(3.)
        w_p.equals_ax_p_by(1./3., u_p, 2.0, v_p)
        zero = w_p.norm2
        if abs(zero) > EPS:
            self.failures['primal_vec']['times'] = True
            self.failures['primal_vec']['equals_ax_p_by'] = True
            self.failures['primal_vec']['inner'] = True
            self.exit_verify = True

        u_p.equals(0.0)
        v_p.exp(u_p)
        w_p.equals(-1.0)
        w_p.plus(v_p)
        zero = w_p.norm2
        if abs(zero) > EPS:
            self.failures['primal_vec']['equals_value'] = True
            self.failures['primal_vec']['exp'] = True
            self.failures['primal_vec']['plus'] = True
            self.failures['primal_vec']['inner'] = True
            self.exit_verify = True

        u_p.equals(1.0)
        v_p.log(u_p)
        zero = v_p.norm2
        if abs(zero) > EPS:
            self.failures['primal_vec']['equals_value'] = True
            self.failures['primal_vec']['log'] = True
            self.failures['primal_vec']['inner'] = True
            self.exit_verify = True

        u_p.equals(2.0)
        u_p.pow(2.0)
        w_p.equals(-4.0)
        w_p.plus(u_p)
        zero = w_p.norm2
        if abs(zero) > EPS:
            self.failures['primal_vec']['equals_value'] = True
            self.failures['primal_vec']['pow'] = True
            self.failures['primal_vec']['plus'] = True
            self.failures['primal_vec']['inner'] = True
            self.exit_verify = True

    def _verify_state_vec(self):
        if not self.optns['state_vec']:
            return

        u_s = self.state_factory.generate()
        v_s = self.state_factory.generate()
        w_s = self.state_factory.generate()

        u_s.equals(10.0)
        u_s.divide_by(u_s.norm2)
        one = u_s.norm2
        if abs(one) - 1 > EPS:
            self.failures['state_vec']['equals_value'] = True
            self.failures['state_vec']['times'] = True
            self.failures['state_vec']['inner'] = True
            self.exit_verify = True

        u_s.equals(1.)
        v_s.equals(-1.)
        w_s.equals(u_s)
        w_s.plus(v_s)
        zero = w_s.norm2
        if abs(zero) > EPS:
            self.failures['state_vec']['equals_value'] = True
            self.failures['state_vec']['equals_vector'] = True
            self.failures['state_vec']['plus'] = True
            self.failures['state_vec']['inner'] = True
            self.exit_verify = True

        u_s.times(2.)
        v_s.divide_by(3.)
        w_s.equals_ax_p_by(1./3., u_s, 2.0, v_s)
        zero = w_s.norm2
        if abs(zero) > EPS:
            self.failures['state_vec']['times'] = True
            self.failures['state_vec']['equals_ax_p_by'] = True
            self.failures['state_vec']['inner'] = True
            self.exit_verify = True

        u_s.equals(0.0)
        v_s.exp(u_s)
        w_s.equals(-1.0)
        w_s.plus(v_s)
        zero = w_s.norm2
        if abs(zero) > EPS:
            self.failures['state_vec']['equals_value'] = True
            self.failures['state_vec']['exp'] = True
            self.failures['state_vec']['plus'] = True
            self.failures['state_vec']['inner'] = True
            self.exit_verify = True

        u_s.equals(1.0)
        v_s.log(u_s)
        zero = v_s.norm2
        if abs(zero) > EPS:
            self.failures['state_vec']['equals_value'] = True
            self.failures['state_vec']['log'] = True
            self.failures['state_vec']['inner'] = True
            self.exit_verify = True

        u_s.equals(2.0)
        u_s.pow(2.0)
        w_s.equals(-4.0)
        w_s.plus(u_s)
        zero = w_s.norm2
        if abs(zero) > EPS:
            self.failures['state_vec']['equals_value'] = True
            self.failures['state_vec']['pow'] = True
            self.failures['state_vec']['plus'] = True
            self.failures['state_vec']['inner'] = True
            self.exit_verify = True

    def _verify_dual_vec(self):
        if not self.optns['dual_vec']:
            return

        u_d = self.dual_factory.generate()
        v_d = self.dual_factory.generate()
        w_d = self.dual_factory.generate()

        u_d.equals(10.0)
        u_d.divide_by(u_d.norm2)
        one = u_d.norm2
        if abs(one) - 1 > EPS:
            self.failures['dual_vec']['equals_value'] = True
            self.failures['dual_vec']['times'] = True
            self.failures['dual_vec']['inner'] = True
            self.exit_verify = True

        u_d.equals(1.)
        v_d.equals(-1.)
        w_d.equals(u_d)
        w_d.plus(v_d)
        zero = w_d.norm2
        if abs(zero) > EPS:
            self.failures['dual_vec']['equals_value'] = True
            self.failures['dual_vec']['equals_vector'] = True
            self.failures['dual_vec']['plus'] = True
            self.failures['dual_vec']['inner'] = True
            self.exit_verify = True

        u_d.times(2.)
        v_d.divide_by(3.)
        w_d.equals_ax_p_by(1./3., u_d, 2.0, v_d)
        zero = w_d.norm2
        if abs(zero) > EPS:
            self.failures['dual_vec']['times'] = True
            self.failures['dual_vec']['equals_ax_p_by'] = True
            self.failures['dual_vec']['inner'] = True
            self.exit_verify = True

        u_d.equals(0.0)
        v_d.exp(u_d)
        w_d.equals(-1.0)
        w_d.plus(v_d)
        zero = w_d.norm2
        if abs(zero) > EPS:
            self.failures['dual_vec']['equals_value'] = True
            self.failures['dual_vec']['exp'] = True
            self.failures['dual_vec']['plus'] = True
            self.failures['dual_vec']['inner'] = True
            self.exit_verify = True

        u_d.equals(1.0)
        v_d.log(u_d)
        zero = v_d.norm2
        if abs(zero) > EPS:
            self.failures['dual_vec']['equals_value'] = True
            self.failures['dual_vec']['log'] = True
            self.failures['dual_vec']['inner'] = True
            self.exit_verify = True

        u_d.equals(2.0)
        u_d.pow(2.0)
        w_d.equals(-4.0)
        w_d.plus(u_d)
        zero = w_d.norm2
        if abs(zero) > EPS:
            self.failures['dual_vec']['equals_value'] = True
            self.failures['dual_vec']['pow'] = True
            self.failures['dual_vec']['plus'] = True
            self.failures['dual_vec']['inner'] = True
            self.exit_verify = True

    def _verify_gradients(self):
        if not self.optns['gradients']:
            return

        u_p = self.primal_factory.generate()
        v_p = self.primal_factory.generate()
        w_p = self.primal_factory.generate()
        z_p = self.primal_factory.generate()
        u_s = self.state_factory.generate()
        v_s = self.state_factory.generate()
        w_s = self.state_factory.generate()
        z_s = self.state_factory.generate()

        u_p.equals_init_design()
        u_s.equals_primal_solution(u_p)
        if self.factor_matrices:
            factor_linear_system(u_p, u_s)
        J = objective_value(u_p, u_s)

        v_p.equals(1./EPS)
        v_p.equals_objective_partial(u_p, u_s)
        z_p.equals(1.0)
        epsilon_fd = calc_epsilon(u_p.norm2, z_p.norm2)
        w_p.equals(z_p)
        w_p.times(epsilon_fd)
        w_p.plus(u_p)
        J_pert = objective_value(w_p, u_s)
        dir_deriv_fd = (J_pert - J)/epsilon_fd
        dir_deriv = v_p.inner(z_p)
        abs_error = abs(dir_deriv - dir_deriv_fd)
        rel_error = abs_error/max(EPS, abs(dir_deriv))

        self.out_stream.write(
            '============================================================\n' +
            'Directional derivative test (design): dF/dX * 1\n' +
            '   FD perturbation      : %e\n'%epsilon_fd +
            '   analytical dir_deriv : %f\n'%dir_deriv +
            '   finite-difference    : %f\n'%dir_deriv_fd +
            '   absolute error       : %e\n'%abs_error +
            '   relative error       : %e\n'%rel_error
        )

        if rel_error > epsilon_fd:
            self.failures['gradients']['eval_dFdX'] = True
            self.out_stream.write(
                'WARNING: eval_dFdX() or eval_obj() may be inaccurate!\n'
            )

        v_s.equals(1./EPS)
        v_s.equals_objective_partial(u_p, u_s)
        z_s.equals(1.0)
        epsilon_fd = calc_epsilon(u_s.norm2, z_s.norm2)
        w_s.equals(z_s)
        w_s.times(epsilon_fd)
        w_s.plus(u_s)
        J_pert = objective_value(u_p, w_s)
        dir_deriv_fd = (J_pert - J)/epsilon_fd
        dir_deriv = v_s.inner(z_s)
        abs_error = abs(dir_deriv - dir_deriv_fd)
        rel_error = abs_error/max(EPS, abs(dir_deriv))

        self.out_stream.write(
            '============================================================\n' +
            'Directional derivative test (state): dF/dU * 1\n' +
            '   FD perturbation      : %e\n'%epsilon_fd +
            '   analytical dir_deriv : %f\n'%dir_deriv +
            '   finite-difference    : %f\n'%dir_deriv_fd +
            '   absolute error       : %e\n'%abs_error +
            '   relative error       : %e\n'%rel_error
        )

        if rel_error > epsilon_fd:
            self.failures['gradients']['eval_dFdU'] = True
            self.out_stream.write(
                'WARNING: eval_dFdU() or eval_obj() may be inaccurate!\n')

    def _verify_pde_jac(self):
        if not self.optns['pde_jac']:
            return

        u_p = self.primal_factory.generate()
        v_p = self.primal_factory.generate()
        w_p = self.primal_factory.generate()
        z_p = self.primal_factory.generate()
        u_s = self.state_factory.generate()
        v_s = self.state_factory.generate()
        w_s = self.state_factory.generate()
        x_s = self.state_factory.generate()
        y_s = self.state_factory.generate()
        z_s = self.state_factory.generate()

        u_p.equals_init_design()
        u_s.equals_primal_solution(u_p)
        if self.factor_matrices:
            factor_linear_system(u_p, u_s)
        z_p.equals(1.0)
        z_s.equals(1.0)
        x_s.equals_residual(u_p, u_s)

        v_s.equals(1./EPS)
        dRdX(u_p, u_s).product(z_p, v_s)
        prod_fwd = v_s.inner(z_s)
        prod_norm = prod_fwd

        epsilon_fd = calc_epsilon(u_p.norm2, z_p.norm2)
        w_p.equals(z_p)
        w_p.times(epsilon_fd)
        w_p.plus(u_p)
        y_s.equals_residual(w_p, u_s)
        y_s.minus(x_s)
        y_s.divide_by(epsilon_fd)
        prod_norm_fd = y_s.inner(z_s)
        v_s.minus(y_s)
        error = abs(v_s.inner(z_s))
        rel_error = error/max(abs(prod_norm_fd), EPS)

        self.out_stream.write(
            '============================================================\n' +
            'PDE jacobian-vector product test (design): dR/dX * 1\n' +
            '   FD perturbation      : %e\n'%epsilon_fd +
            '   analytical product   : %f\n'%prod_norm +
            '   FD product           : %f\n'%prod_norm_fd +
            '   absolute error       : %e\n'%error +
            '   relative error       : %e\n'%rel_error
        )
        if rel_error > epsilon_fd:
            self.failures['pde_jac']['multiply_dRdX'] = True
            self.out_stream.write(
                'WARNING: multiply_dRdX or eval_residual may be inaccurate!\n'
            )

        v_p.equals(1./EPS)
        dRdX(u_p, u_s).T.product(z_s, v_p)
        prod_rev = v_p.inner(z_p)
        abs_error = abs(prod_fwd - prod_rev)
        rel_error = abs_error/max(abs(prod_fwd), EPS)

        self.out_stream.write(
            '============================================================\n' +
            'PDE jacobian-vector transpose-product test (design): \n' +
            '1^{T} dR/dX * 1\n' +
            '   forward product      : %f\n'%prod_fwd +
            '   reverse product      : %f\n'%prod_rev +
            '   absolute error       : %e\n'%abs_error +
            '   relative error       : %e\n'%rel_error
        )

        if rel_error > epsilon_fd:
            self.failures['pde_jac']['multiply_dRdX_T'] = True
            self.out_stream.write(
                'WARNING: multiply_dRdX_T() may be inaccurate!\n'
            )
            if self.failures['pde_jac']['multiply_dRdX']:
                self.out_stream.write(
                    'WARNING: Fix multiply_dRdX() and check this test again!\n'
                )

        v_s.equals(1./EPS)
        z_s.equals(1.)
        dRdU(u_p, u_s).product(z_s, v_s)
        prod_norm = v_s.inner(z_s)

        epsilon_fd = calc_epsilon(u_s.norm2, z_s.norm2)
        w_s.equals(z_s)
        w_s.times(epsilon_fd)
        w_s.plus(u_s)
        y_s.equals_residual(u_p, w_s)
        y_s.minus(x_s)
        y_s.divide_by(epsilon_fd)
        prod_fd = y_s.inner(z_s)
        prod_norm_fd = prod_fd
        v_s.minus(y_s)
        error = abs(v_s.inner(z_s))
        rel_error = error/max(abs(prod_norm_fd), EPS)

        self.out_stream.write(
            '============================================================\n' +
            'PDE jacobian-vector product test (state): dR/dU * 1\n' +
            '   FD perturbation      : %e\n'%epsilon_fd +
            '   analytical product   : %f\n'%prod_norm +
            '   FD product           : %f\n'%prod_norm_fd +
            '   absolute error       : %e\n'%error +
            '   relative error       : %e\n'%rel_error
        )

        if rel_error > epsilon_fd:
            self.failures['pde_jac']['multiply_dRdU'] = True
            self.out_stream.write(
                'WARNING: multiply_dRdU() or eval_residual() ' +
                'may be inaccurate!\n'
            )

        v_s.equals(1./EPS)
        dRdU(u_p, u_s).T.product(z_s, v_s)
        prod = v_s.inner(z_s)
        abs_error = abs(prod - prod_fd)
        rel_error = abs_error/max(abs(prod), EPS)

        self.out_stream.write(
            '============================================================\n' +
            'PDE jacobian-vector transpose-product test (state): \n' +
            '1^{T} dR/dU * 1\n' +
            '   FD perturbation      : %e\n'%epsilon_fd +
            '   analytical product   : %f\n'%prod +
            '   FD product           : %f\n'%prod_fd +
            '   absolute error       : %e\n'%abs_error +
            '   relative error       : %e\n'%rel_error
        )

        if rel_error > epsilon_fd:
            self.failures['pde_jac']['multiply_dRdU_T'] = True
            self.out_stream.write(
                'WARNING: multiply_dRdU_T() may be inaccurate!\n'
            )
            if self.failures['pde_jac']['multiply_dRdU']:
                self.out_stream.write(
                    'WARNING: Fix multiply_dRdU() and check this test again!\n'
                )

    def _verify_cnstr_jac(self):
        if not self.optns['cnstr_jac']:
            return

        u_p = self.primal_factory.generate()
        v_p = self.primal_factory.generate()
        w_p = self.primal_factory.generate()
        z_p = self.primal_factory.generate()
        u_s = self.state_factory.generate()
        v_s = self.state_factory.generate()
        w_s = self.state_factory.generate()
        z_s = self.state_factory.generate()
        v_d = self.dual_factory.generate()
        x_d = self.dual_factory.generate()
        y_d = self.dual_factory.generate()
        z_d = self.dual_factory.generate()

        u_p.equals_init_design()
        u_s.equals_primal_solution(u_p)
        if self.factor_matrices:
            factor_linear_system(u_p, u_s)
        x_d.equals_constraints(u_p, u_s)

        z_p.equals(1.0)
        z_d.equals(1.0)
        v_d.equals(1./EPS)
        dCdX(u_p, u_s).product(z_p, v_d)
        prod_norm = v_d.inner(z_d)

        epsilon_fd = calc_epsilon(u_p.norm2, z_p.norm2)
        w_p.equals(z_p)
        w_p.times(epsilon_fd)
        w_p.plus(u_p)
        y_d.equals_constraints(w_p, u_s)
        y_d.minus(x_d)
        y_d.divide_by(epsilon_fd)
        prod_norm_fd = y_d.inner(z_d)
        v_d.minus(y_d)
        error = abs(v_d.inner(z_d))
        rel_error = error/max(abs(prod_norm), EPS)

        self.out_stream.write(
            '============================================================\n' +
            'Constraint jacobian-vector product test (design):\n' +
            'dC/dX * 1\n' +
            '   FD perturbation      : %e\n'%epsilon_fd +
            '   analytical product   : %f\n'%prod_norm +
            '   FD product           : %f\n'%prod_norm_fd +
            '   absolute error       : %e\n'%error +
            '   relative error       : %e\n'%rel_error
        )

        if rel_error > epsilon_fd:
            self.failures['cnstr_jac']['multiply_dCdX'] = True
            self.out_stream.write(
                'WARNING: multiply_dCdX() or eval_constraints() ' +
                'may be inaccurate!\n'
            )

        z_d.equals(1.0)
        v_p.equals(1./EPS)
        dCdX(u_p, u_s).T.product(z_d, v_p)
        prod = v_p.inner(z_p)
        prod_fd = y_d.inner(z_d)
        abs_error = abs(prod - prod_fd)
        rel_error = abs_error/max(abs(prod), EPS)

        self.out_stream.write(
            '============================================================\n' +
            'Constraint jacobian-vector transpose-product test (design): \n' +
            '1^{T} dC/dX * 1\n' +
            '   FD perturbation      : %e\n'%epsilon_fd +
            '   analytical product   : %f\n'%prod +
            '   FD product           : %f\n'%prod_fd +
            '   absolute error       : %e\n'%abs_error +
            '   relative error       : %e\n'%rel_error
        )

        if rel_error > epsilon_fd:
            self.failures['cnstr_jac']['multiply_dCdX_T'] = True
            self.out_stream.write(
                'WARNING: multiply_dCdX_T() may be inaccurate!\n'
            )
            if self.failures['cnstr_jac']['multiply_dCdX']:
                self.out_stream.write(
                    'WARNING: Fix multiply_dCdX() and check this test again!\n'
                )

        z_s.equals(1.0)
        z_d.equals(1.0)
        v_d.equals(1./EPS)
        dCdU(u_p, u_s).product(z_s, v_d)
        prod_norm = v_d.inner(z_d)
        epsilon_fd = calc_epsilon(u_s.norm2, z_s.norm2)
        w_s.equals(z_s)
        w_s.times(epsilon_fd)
        w_s.plus(u_s)
        y_d.equals_constraints(u_p, w_s)
        y_d.minus(x_d)
        y_d.divide_by(epsilon_fd)
        prod_norm_fd = y_d.inner(z_d)
        v_d.minus(y_d)
        error = abs(v_d.inner(z_d))
        rel_error = error/max(abs(prod_norm), EPS)

        self.out_stream.write(
            '============================================================\n' +
            'Constraint jacobian-vector product test (state):\n' +
            'dC/dU * 1\n' +
            '   FD perturbation      : %e\n'%epsilon_fd +
            '   analytical product   : %f\n'%prod_norm +
            '   FD product           : %f\n'%prod_norm_fd +
            '   absolute error       : %e\n'%error +
            '   relative error       : %e\n'%rel_error
        )

        if rel_error > epsilon_fd:
            self.failures['cnstr_jac']['multiply_dCdU'] = True
            self.out_stream.write(
                'WARNING: multiply_dCdU() or eval_constraints() ' +
                'may be inaccurate!\n'
            )

        v_s.equals(1./EPS)
        dCdU(u_p, u_s).T.product(z_d, v_s)
        prod = v_s.inner(z_s)
        prod_fd = y_d.inner(z_d)
        abs_error = abs(prod - prod_fd)
        rel_error = abs_error/max(abs(prod), EPS)

        self.out_stream.write(
            '============================================================\n' +
            'Constraint jacobian-vector transpose-product test (state): \n' +
            '1^{T} dC/dU * 1\n' +
            '   FD perturbation      : %e\n'%epsilon_fd +
            '   analytical product   : %f\n'%prod +
            '   FD product           : %f\n'%prod_fd +
            '   absolute error       : %e\n'%abs_error +
            '   relative error       : %e\n'%rel_error
        )

        if rel_error > epsilon_fd:
            self.failures['cnstr_jac']['multiply_dCdU_T'] = True
            self.out_stream.write(
                'WARNING: multiply_dCdU_T() may be inaccurate!\n'
            )
            if self.failures['cnstr_jac']['multiply_dCdU']:
                self.out_stream.write(
                    'WARNING: Fix multiply_dCdU() and check this test again!\n'
                )

    def _verify_red_grad(self):
        if not self.optns['red_grad']:
            return

        u_p = self.primal_factory.generate()
        v_p = self.primal_factory.generate()
        w_p = self.primal_factory.generate()
        z_p = self.primal_factory.generate()
        u_s = self.state_factory.generate()
        v_s = self.state_factory.generate()
        w_s = self.state_factory.generate()

        u_p.equals_init_design()
        u_s.equals_primal_solution(u_p)
        if self.factor_matrices:
            factor_linear_system(u_p, u_s)
        J = objective_value(u_p, u_s)

        v_s.equals_objective_partial(u_p, u_s)
        v_s.equals_adjoint_solution(u_p, u_s, w_s)
        v_p.equals_total_gradient(u_p, u_s, v_s, w_p)
        z_p.equals(1.0)
        prod = z_p.inner(v_p)

        epsilon_fd = calc_epsilon(u_p.norm2, z_p.norm2)
        w_p.equals(z_p)
        w_p.times(epsilon_fd)
        w_p.plus(u_p)
        w_s.equals_primal_solution(w_p)
        if self.factor_matrices:
            factor_linear_system(w_p, w_s)
        J_pert = objective_value(w_p, w_s)
        prod_fd = (J_pert - J)/epsilon_fd
        abs_error = abs(prod - prod_fd)
        rel_error = abs_error/max(abs(prod), EPS)

        self.out_stream.write(
            '============================================================\n' +
            'Reduced gradient (total derivative) test: dF/dX * 1 \n' +
            '   FD perturbation      : %e\n'%epsilon_fd +
            '   analytical product   : %f\n'%prod +
            '   FD product           : %f\n'%prod_fd +
            '   absolute error       : %e\n'%abs_error +
            '   relative error       : %e\n'%rel_error
        )

        if rel_error > epsilon_fd:
            self.failures['red_grad']['solve_adjoint'] = True
            self.out_stream.write(
                'WARNING: solve_adjoint() may be inaccurate!\n'
            )
            if self.failures['gradients']['eval_dFdX']:
                self.out_stream.write(
                    'WARNING: Fix eval_dFdX() and check this test again!\n'
                )
            if self.failures['gradients']['eval_dFdU']:
                self.out_stream.write(
                    'WARNING: Fix eval_dFdU() and check this test again!\n'
                )
            if self.failures['pde_jac']['multiply_dRdX_T']:
                self.out_stream.write(
                    'WARNING: Fix multiply_dRdX_T() and check this again!\n'
                )

    def _verify_lin_solve(self):
        if not self.optns['lin_solve']:
            return

        primal = self.primal_factory.generate()
        primal_sol = self.state_factory.generate()
        u = self.state_factory.generate()
        v = self.state_factory.generate()
        w = self.state_factory.generate()
        z = self.state_factory.generate()

        primal.equals_init_design()
        primal_sol.equals_primal_solution(primal)
        if self.factor_matrices:
            factor_linear_system(primal, primal_sol)

        u.equals_objective_partial(primal, primal_sol)
        v.equals(u)
        rel_tol = 1e-6

        dRdU(primal, primal_sol).solve(u, w, rel_tol)
        forward = v.inner(w)
        dRdU(primal, primal_sol).T.solve(v, z, rel_tol)
        reverse = u.inner(z)
        error = forward - reverse

        self.out_stream.write(
            '============================================================\n' +
            'Linear solve test: dR/dU * w = z \n' +
            '   FWD solve product    : %e\n'%forward +
            '   REV solve product    : %e\n'%reverse +
            '   difference           : %e\n'%error
        )

        if abs(forward - reverse) > (abs(forward) + EPS)*10.0*rel_tol:
            self.failures['lin_solve']['solve_linear'] = True
            self.out_stream.write(
                'WARNING: solve_linear() may be inaccurate!\n'
            )
            if self.failures['red_grad']['solve_adjoint']:
                self.out_stream.write(
                    'WARNING: Fix solve_adjoint() and check this test again!\n'
                )
