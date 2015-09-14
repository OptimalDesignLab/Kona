from rosenbrock import Rosenbrock
from simple_constrained import SimpleConstrained
from simple_2by2 import Simple2x2
from constrained_2by2 import Constrained2x2
from poisson import InversePoisson
from spiral import Spiral

try:
    from mdo_idf import ScalableIDF
except ImportError:
    print 'WARNING: PyAMG not installed. ScalableIDF example will not work.'

# try:
#     from laplace2d import Laplace2D
# except ImportError:
#     print 'WARNING: Scipy or matplotlib not installed. Laplace2D example will not work.'
