import unittest

from kona.linalg.memory import KonaMemory
from kona.linalg import objective_value
from kona.linalg.vectors.composite import ReducedKKTVector
from kona.linalg.vectors.composite import CompositePrimalVector

from kona.algorithms.util.merit import AugmentedLagrangian
from kona.examples import Sellar

class AugLagMeritTestCase(unittest.TestCase):

    def setUp(self):
        solver = Sellar()
        self.km = KonaMemory(solver)
        self.pf = self.km.primal_factory
        self.sf = self.km.state_factory
        self.df = self.km.dual_factory

        self.pf.request_num_vectors(10)
        self.sf.request_num_vectors(10)
        self.df.request_num_vectors(10)
        optns = {'freeze_mults': True}
        self.merit = AugmentedLagrangian(self.pf, self.sf, self.df, optns)

        self.km.allocate_memory()

        self.mu = 10.0

        self.kkt_start = ReducedKKTVector(
            CompositePrimalVector(self.pf.generate(), self.df.generate()),
            self.df.generate())
        self.kkt_trial = ReducedKKTVector(
            CompositePrimalVector(self.pf.generate(), self.df.generate()),
            self.df.generate())
        self.search_dir = ReducedKKTVector(
            CompositePrimalVector(self.pf.generate(), self.df.generate()),
            self.df.generate())
        self.u_start = self.sf.generate()
        self.u_trial = self.sf.generate()
        self.cnstr = self.df.generate()
        self.cnstr_trial = self.df.generate()
        self.slack_term = self.df.generate()

        self.search_dir.equals(1.0)

        self.kkt_start.equals_init_guess()
        self.kkt_start._primal._design.enforce_bounds()

        self.u_start.equals_primal_solution(self.kkt_start._primal._design)

        self.cnstr.equals_constraints(
            self.kkt_start._primal._design, self.u_start)
        self.slack_term.exp(self.kkt_start._primal._slack)
        self.slack_term.restrict()
        self.cnstr.minus(self.slack_term)

        obj_val = objective_value(
            self.kkt_start._primal._design, self.u_start)
        aug_term = self.kkt_start._dual.inner(self.cnstr)
        penalty_term = 0.5*self.mu*(self.cnstr.norm2**2)
        self.merit_val_init = obj_val + aug_term + penalty_term

    def test_init_func(self):
        self.merit.reset(self.kkt_start, self.u_start, self.search_dir, self.mu)
        self.assertEqual(self.merit.func_val, self.merit_val_init)

    def test_eval_func(self):
        # get merit value from merit object
        self.merit.reset(self.kkt_start, self.u_start, self.search_dir, self.mu)
        alpha = 1.
        merit_value = self.merit.eval_func(alpha)

        # calculate expected merit value
        self.kkt_trial.equals_ax_p_by(
            1., self.kkt_start, alpha, self.search_dir)
        self.kkt_trial._primal._design.enforce_bounds()
        self.u_trial.equals_primal_solution(self.kkt_trial._primal._design)
        self.cnstr_trial.equals_constraints(
            self.kkt_trial._primal._design, self.u_trial)
        self.slack_term.exp(self.kkt_trial._primal._slack)
        self.slack_term.restrict()
        self.cnstr_trial.minus(self.slack_term)
        obj_val = objective_value(
            self.kkt_trial._primal._design, self.u_trial)
        aug_term = self.kkt_start._dual.inner(self.cnstr_trial)
        penalty_term = 0.5*self.mu*(self.cnstr_trial.norm2**2)
        expected_value = obj_val + aug_term + penalty_term

        # compare the two
        self.assertEqual(merit_value, expected_value)

    def test_unnecessary_func_eval(self):
        self.merit.reset(self.kkt_start, self.u_start, self.search_dir, self.mu)
        expected_value = self.merit.func_val
        init_alpha = self.merit.last_func_alpha
        alpha = init_alpha + 1e-20
        merit_value = self.merit.eval_func(alpha)
        self.assertEqual(merit_value, expected_value)

if __name__ == "__main__":

    unittest.main()
