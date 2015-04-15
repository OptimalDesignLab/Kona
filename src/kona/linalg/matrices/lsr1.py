from kona.linalg.matrices.quasi_newton import QuasiNewtonApprox
from kona.options import get_opt
import numpy
import sys, gc


class LimitedMemorySR1(QuasiNewtonApprox):
    """ Limited memory symmetric rank-one update

    Attributes
    ----------
    lambda0 : float
        ?
    """

    def __init__(self, vector_factory, optns, out_file=sys.stdout):
        super(LimitedMemorySR1, self).__init__(vector_factory, optns, out_file)

        self.lambda0 = 0

        self.max_stored = get_opt(optns, 10, 'max_stored')

        vector_factory.request_num_vectors(3*self.max_stored + 4)

    def _hessian_product(self, u_vec, v_vec):
        s_list = self.s_list
        y_list = self.y_list
        num_stored = len(s_list)

        Bs = []
        for k in xrange(num_stored):
            Bs.append(self.vec_fac.generate())
            Bs[k].equals(s_list[k])

        v_vec.equals(u_vec)

        for i in xrange(num_stored):
            denom = 1.0 / (y_list[i].inner(s_list[i]) - Bs[i].inner(s_list[i]))
            fac = (y_list[i].inner(u_vec) - Bs[i].inner(u_vec)) * denom
            v_vec.equals_ax_p_by(1.0, v_vec, fac, y_list[i])
            v_vec.equals_ax_p_by(1.0, v_vec, -fac, Bs[i])
            for j in xrange(i+1, num_stored):
                fac = (y_list[i].inner(s_list[j]) - Bs[i].inner(s_list[j])) * denom
                Bs[j].equals_ax_p_by(1.0, Bs[j], fac, y_list[i])
                Bs[j].equals_ax_p_by(1.0, Bs[j], -fac, Bs[i])

    def add_correction(self, s_in, y_in):
        """
        Add the step and change in gradient to the lists storing the history.
        """
        threshold = 1.e-8

        y_copy = self.vec_fac.generate()
        tmp = self.vec_fac.generate()
        y_copy.equals(y_in)
        self._hessian_product(y_copy, tmp)
        tmp.minus(s_in)

        norm_resid = tmp.norm2
        norm_y_new = y_in.norm2
        prod = abs(y_in.inner(tmp))

        if prod < threshold * norm_resid * norm_y_new or \
                prod < numpy.finfo(float).eps:
            self.out_file.write('LimitedMemorySR1::AddCorrection():' +
                               'correction skipped due to threshold condition.')
            return

        if len(self.s_list) == self.max_stored:
            del self.s_list[0]
            del self.y_list[0]

        s_new = self.vec_fac.generate()
        y_new = self.vec_fac.generate()

        s_new.equals(s_in)
        y_new.equals(y_in)

        self.s_list.append(s_new)
        self.y_list.append(y_new)

        del s_new, y_new
        gc.collect()

    def solve(self, u_vec, v_vec, rel_tol=1e-15):
        lambda0 = self.lambda0
        norm_init = self.norm_init
        s_list = self.s_list
        y_list = self.y_list

        num_stored = len(s_list)

        if num_stored == 0:
            v_vec.equals(u_vec)
            v_vec.divide_by(norm_init)
            return

        threshold = 1.e-8
        alpha = numpy.zeros(num_stored)

        z_list = []
        for k in xrange(num_stored):
            z_list.append(self.vec_fac.generate())
            z_list[k].equals_ax_p_by(1.0 - lambda0, s_list[k],
                                     (lambda0 - 1.0)/norm_init,
                                     y_list[k])

        def check_threshold(self, k, alpha):
            lambda0 = self.lambda0
            norm_init = self.norm_init
            s_list = self.s_list
            y_list = self.y_list

            alpha[k] = (1.0 - lambda0) * z_list[k].inner(y_list[k]) + \
                lambda0 * norm_init * z_list[k].inner(s_list[k])
            norm_grad = (1.0 - lambda0) ** 2 * y_list[k].inner(y_list[k]) + \
                lambda0 ** 2 * norm_init ** 2 * s_list[k].inner(s_list[k]) + \
                2.0 * lambda0 * norm_init * (1.0 - lambda0) * y_list[k].inner(s_list[k])
            norm_grad = numpy.sqrt(norm_grad)
            if abs(alpha[k]) < threshold * norm_grad * z_list[k].norm2 or \
                    abs(alpha[k]) < numpy.finfo(float).eps:
                alpha[k] = 0.0
            else:
                alpha[k] = 1.0 / alpha[k]

        check_threshold(self, 0, alpha)

        for i in xrange(1, num_stored):
            for j in xrange(1, num_stored):
                prod = (1.0 - lambda0) * z_list[i-1].inner(y_list[j]) + \
                    lambda0 * norm_init * z_list[i-1].inner(s_list[j])
                z_list[j].equals_ax_p_by(1.0, z_list[j],
                                         -alpha[i-1] * prod, z_list[i-1])

            check_threshold(self, i, alpha)

        v_vec.equals(u_vec)
        for k in xrange(num_stored):
            v_vec.equals_ax_p_by(1.0, v_vec,
                                 alpha[k] * z_list[k].inner(u_vec), z_list[k])
