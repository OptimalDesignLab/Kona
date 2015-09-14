# import numpy as np
# from kona.examples import Laplace2D
#
# problem = Laplace2D(1., 1., 100, 100)
#
# problem.set_boundary('dirichlet', 'west', np.ones(100))
# problem.set_boundary('dirichlet', 'east', np.ones(100))
# problem.set_boundary('neumann', 'north', 1.0)
# problem.set_boundary('neumann', 'south', -1.0)
#
# problem.direct_solve()
#
# problem.plot_field(save=True)
