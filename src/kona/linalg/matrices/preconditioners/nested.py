from kona.linalg.matrices.hessian.basic import BaseHessian

class NestedKKTPreconditioner(BaseHessian):
    """
    This is a very simple wrapper around the ReducedKKTMatrix object. It
    performs an approximate/exact solve of the KKT system using user solver
    dRdU preconditioners.

    Parameters
    ----------
    KKT_matrix : ReducedKKTMatrix
        The KKT matrix to be preconditioned.
    krylov_solver : KrylovSolver
        Krylov solver used in the nested solution.

    Attributes
    ----------
    KKT_matrix : ReducedKKTMatrix
    """
    def __init__(self, KKT_matrix, krylov_solver):
        self.KKT_matrix = KKT_matrix
        self.KKT_matrix.set_krylov_solver(krylov_solver)

    def product(self, in_vec, out_vec):
        self.KKT_matrix.approx.solve(in_vec, out_vec)
