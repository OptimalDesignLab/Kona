import sys
from numpy import sqrt

from kona.options import BadKonaOption, get_opt

from kona.linalg import current_solution, objective_value
from kona.linalg.matrices.common import dRdX, dRdU, IdentityMatrix
from kona.linalg.matrices.hessian import LimitedMemoryBFGS, ReducedHessian
from kona.linalg.solvers.krylov import STCG

from kona.algorithms.base_algorithm import OptimizationAlgorithm
from kona.algorithms.util.merit import ObjectiveMerit
from kona.algorithms.util.linesearch import StrongWolfe

class ReducedSpaceNewtonKrylov(OptimizationAlgorithm):
    """
    Optimization using a Newton-Krylov approach in the reduced space.

    """
    def __init__(self, primal_factory, state_factory, optns={}):
        # trigger base class initialization
        super(ReducedSpaceNewtonKrylov, self).__init__(
            primal_factory, state_factory, None, optns
        )

        # number of vectors required in solve() method
        self.primal_factory.request_num_vectors(7)
        self.state_factory.request_num_vectors(8)

        # set the type of merit function
        self.merit_func = ObjectiveMerit(self.primal_factory, self.state_factory)

        # get other options
        self.radius = get_opt(optns, 0.1, 'trust', 'init_radius')
        self.max_radius = get_opt(optns, 1.0, 'trust', 'max_radius')
        self.trust_tol = get_opt(optns, 0.1, 'trust', 'tol')

        # set the type of quasi-Newton method
        try:
            quasi_newton = get_opt(optns, LimitedMemoryBFGS, 'quasi_newton', 'type')
            qn_optns = get_opt(optns, {}, 'quasi_newton')
            self.quasi_newton = quasi_newton(self.primal_factory, qn_optns)
        except Exception as err:
            raise BadKonaOption(optns, 'quasi_newton','type')

        # set the krylov solver
        try:
            krylov = get_opt(optns, STCG, 'krylov', 'solver')
            krylov_optns = get_opt(optns, {}, 'krylov')
            self.krylov = krylov(self.primal_factory, krylov_optns)
            self.krylov.radius = self.radius
        except:
            raise BadKonaOption(optns, 'krylov', 'solver')

        # initialize the ReducedHessian approximation
        reduced_optns = get_opt(optns, {}, 'reduced')
        self.hessian = ReducedHessian(
            [self.primal_factory, self.state_factory], reduced_optns)
        reduced_precond = get_opt(optns, None, 'reduced', 'precond')
        if reduced_precond is 'quasi_newton':
            self.hessian.quasi_newton = self.quasi_newton
        self.hessian.set_krylov_solver(self.krylov)

    def _write_header(self):
        self.hist_file.write(
            '# Kona trust-region RSNK convergence history file\n' + \
            '# iters' + ' '*5 + \
            '      cost' + ' '*5 + \
            ' grad norm' + ' '*5 + \
            '     ratio' + ' '*5 + \
            '    radius' + '\n'
        )

    def _write_history(self, num_iter, norm, rho):
        self.hist_file.write(
            ' %6i'%num_iter + ' '*5 + \
            '%10e'%self.primal_factory._memory.cost + ' '*5 + \
            '%10e'%norm + ' '*5 + \
            '%10e'%rho + ' '*5 + \
            '%10e'%self.radius + '\n'
        )

    def solve(self):

        x = self.primal_factory.generate()
        p = self.primal_factory.generate()
        dJdX = self.primal_factory.generate()
        dJdX_old = self.primal_factory.generate()
        primal_work = self.primal_factory.generate()
        state = self.state_factory.generate()
        adjoint = self.state_factory.generate()
        adjoint_old = self.state_factory.generate()
        state_work = self.state_factory.generate()

        # set initial design and solve for state
        x.equals_init_design()
        state.equals_primal_solution(x)
        # initialize design and state jacobians
        design_jac = dRdX(x, state)
        state_jac = dRdU(x, state)
        # solve for adjoint
        adjoint.equals_adjoint_solution(x, state, state_work)
        # get objective value
        obj = objective_value(x, state)

        # START THE NEWTON LOOP
        #######################
        self._write_header()
        nonlinear_sum = 0
        rho = 0.0
        for i in xrange(self.max_iter):

            self.info_file.write('==================================================\n')
            self.info_file.write('Beginning Trust-Region iteration\n')
            self.info_file.write('\n')

            # update quasi-newton
            dJdX.equals_total_gradient(x, state, adjoint, primal_work)
            if i == 0:
                grad_norm0 = dJdX.norm2
                grad_norm = grad_norm0
                self.info_file.write('grad_norm0 = %e\n'%grad_norm0)
                grad_tol = self.primal_tol*grad_norm0
                dJdX_old.equals(dJdX)
            else:
                grad_norm = dJdX.norm2
                self.info_file.write('grad norm : grad_tol = %e : %e\n'%(grad_norm, grad_tol))
                dJdX_old.minus(dJdX)
                dJdX_old.times(-1.0)
                self.quasi_newton.add_correction(p, dJdX_old)
                dJdX_old.equals(dJdX)
            # write history
            current_solution(x, state, adjoint, num_iter=nonlinear_sum)
            self._write_history(nonlinear_sum, grad_norm, rho)
            # check convergence
            if grad_norm < grad_tol:
                converged = True
                break

            krylov_tol = self.krylov.rel_tol*min(1.0, sqrt(grad_norm/grad_norm0))
            krylov_tol = max(krylov_tol, grad_tol/grad_norm)
            krylov_tol *= self.hessian.nu

            self.info_file.write('krylov tol = %e\n'%krylov_tol)
            if self.hessian.dynamic_tol:
                self.hessian.product_fac *= krylov_tol/self.krylov.max_iter

            # inexactly solve the trust-region problem
            p.equals(0.0)
            dJdX.times(-1.0)
            self.krylov.rel_tol = krylov_tol
            self.krylov.radius = self.radius
            self.hessian.linearize(x, state, adjoint)
            pred, active = self.krylov.solve(self.hessian.product, dJdX, p, self.quasi_newton.solve)
            dJdX.times(-1.0)
            x.plus(p)

            # compute the actual reduction and trust parameter rho
            obj_old = obj
            state_work.equals(state)
            if state.equals_primal_solution(x):
                obj = objective_value(x, state)
                rho = (obj_old - obj)/pred
            else:
                rho = -1e-16

            # update radius if necessary
            if rho < 0.25:
                self.radius *= 0.25
            else:
                if active and (rho > 0.75):
                    self.radius = min(2*self.radius, self.max_radius)

            # revert the solution if necessary
            if rho < self.trust_tol:
                self.info_file.write('reverting solution...\n')
                x.minus(p)
                state.equals(state_work)
            else:
                adjoint.equals_adjoint_solution(x, state, state_work)

            nonlinear_sum += 1
        #####################
        # END THE NEWTON LOOP

        self.info_file.write('Total number of nonlinear iterations: %i\n'%nonlinear_sum)
