import numpy as np
import scipy.sparse as sp
import numbers

import matplotlib
matplotlib.use('TkAgg')
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

class Laplace2D(object):
    """
    Implicit Finite Difference solver for the 2D Laplace equations. Uses a
    5-point stencil on an L-by-L Cartesian grid. Implements ghost-point method
    for Neumann boundary conditions.

    This solver can handle both Dirichlet and Neumann boundary conditions on
    any domain edge. However, it cannot handle mixed (Robin) boundary
    conditions.

    Parameters
    ----------
    lx, ly: float
        X and Y length of the rectangular domain
    nx, ny: int
        Number of discrete points along the X and Y direction
    """
    def __init__(self, lx, ly, nx, ny, source=None, mg_levels=0):

        # problem sizing
        self.nx = nx
        self.lx = lx
        self.hx = lx/(nx-1)
        self.ny = ny
        self.ly = ly
        self.hy = ly/(ny-1)

        # dirichlet BC dictionary
        self.dirichlet = {
            'north' : None,
            'south' : None,
            'east'  : None,
            'west'  : None,
        }
        # neumann BC dictionary
        self.neumann = {
            'north' : None,
            'south' : None,
            'east'  : None,
            'west'  : None,
        }

        # allocate the source matrix
        if source is not None:
            if source.shape == (self.nx-2, self.ny-2):
                self.source_terms = np.array(source)
            else:
                raise ValueError('Source term is not the correct shape!')
        else:
            self.source_terms = np.zeros((self.nx-2, self.ny-2))
        # allocate boundary terms
        self.bc_terms = np.zeros((self.nx-2, self.ny-2))
        # allocate the solution
        self.solution = np.zeros((self.nx, self.ny))

        # internal flags
        self.rhs_built = False
        self.coeff_built = False

    def set_source(self, source):
        """
        Set the (nx-2) by (ny-2) source term for the domain.

        Parameters
        ----------
        source : array-like
            (nx-2) by (ny-2) array of source values.
        """
        if source.shape == (self.nx-2, self.ny-2):
            self.source_terms = np.array(source)
        else:
            raise ValueError('Source term is not the correct shape!')

        self.rhs_built = False

    def set_boundary(self, kind, edge, value):
        """
        Set Dirichlet or Neumann boundary conditions for the edges.

        Parameters
        ----------
        kind : str
            BC type -- can be 'dirichlet' or 'neumann'
        edge : str
            Edge designation -- 'north', 'south', 'east' or 'west'
        value : np.Real or list of np.Real
            BC value, must be a scalar for Neumann BCs

        Returns
        -------
        np.ndarray : (nx-2) by (ny-2) dense array of BC terms
        """
        # make sure the BC type is valid
        if kind not in ['dirichlet', 'neumann']:
            raise TypeError('Unrecognized BC type!')
        # make sure the edge string is valid
        if edge not in ['north', 'south', 'east', 'west']:
            raise TypeError('Unrecognized edge name!')

        # set Dirichlet BC values for the edge
        if kind == 'dirichlet':
            # if value is None, we don't need any fancy checks
            if value is None:
                self.dirichlet[edge] = None
                return

            # complain if there's already a Neumann BC
            if self.neumann[edge] is not None:
                raise ValueError('Edge has Neumann BC. Cannot set Dirichlet!')

            # check x or y directionality and save correctly sized BCs
            if edge in ['east', 'west']:
                size = self.ny
            elif edge in ['north', 'south']:
                size = self.nx

            # check if value is scalar or an array
            if isinstance(value, numbers.Real):
                self.dirichlet[edge] = value*np.ones(size)
                return
            elif len(value) == self.ny:
                # make sure to save it as a numpy array
                self.dirichlet[edge] = np.array(value)
                return
            else:
                raise ValueError('Invalid BC value!')

        # set Neumann BC values for the edge
        elif kind == 'neumann':
            # if value is None, we don't need any fancy checks
            if value is None:
                self.dirichlet[edge] = None
                return

            # complain if there's already a Dirichlet BC
            if self.dirichlet[edge] is not None:
                raise ValueError('Edge has Dirichlet BC. Cannot set Neumann!')

            # make sure the BC is a scalar value
            if isinstance(value, numbers.Real):
                self.neumann[edge] = value
                return
            else:
                raise ValueError('Neumann BC value must be a scalar!')

        self.rhs_built = False

    def calculate_boundary_terms(self):
        """
        Processes the boundary values defined on domain edges into a dense
        array that can be applied to the source term.
        """
        # Dirichlet BCs:
        if self.dirichlet['north'] is not None:
            u_north = self.dirichlet['north']
            self.bc_terms[:, -1] = u_north[1:-1]/self.hy**2
        else:
            u_north = np.zeros(self.nx)

        if self.dirichlet['south'] is not None:
            u_south = self.dirichlet['south']
            self.bc_terms[:, 0] = u_south[1:-1]/self.hy**2
        else:
            u_south = np.zeros(self.nx)

        if self.dirichlet['east'] is not None:
            u_east = self.dirichlet['east']
            self.bc_terms[-1, :] = u_east[1:-1]/self.hx**2
        else:
            u_east = np.zeros(self.ny)

        if self.dirichlet['west'] is not None:
            u_west = self.dirichlet['west']
            self.bc_terms[0, :] = u_west[1:-1]/self.hx**2
        else:
            u_west = np.zeros(self.ny)

        # Neumann BCs:
        if self.neumann['north'] is not None:
            du_north = self.neumann['north']
            self.bc_terms[:, -1] = du_north/self.hy
        else:
            du_north = 0.0

        if self.neumann['south'] is not None:
            du_south = self.neumann['north']
            self.bc_terms[:, 0] = -du_south/self.hy
        else:
            du_south = 0.0

        if self.neumann['east'] is not None:
            du_east = self.neumann['east']
            self.bc_terms[-1, :] = du_east/self.hx
        else:
            du_east = 0.0

        if self.neumann['west'] is not None:
            du_west = self.neumann['west']
            self.bc_terms[0, :] = -du_west/self.hx
        else:
            du_west = 0.0

        # BCs at corner nodes:
        self.bc_terms[0, 0] = u_west[1]/self.hx**2 - du_south/self.hy
        self.bc_terms[-1, 0] = u_east[1]/self.hx**2 - du_south/self.hy
        self.bc_terms[0, -1] = u_west[-2]/self.hx**2 + du_north/self.hy
        self.bc_terms[-1, -1] = u_east[-2]/self.hx**2 + du_north/self.hy

    def build_coeff_matrix(self):
        """
        Computes the coefficient matrix and the source vector based on the BCs.
        """
        # (nx-2) by (nx-2) interior point [1, -2, 1]  banded matrix
        data = np.ones(self.nx-3)
        row = np.r_[1:self.nx-2]
        col = np.r_[0:self.nx-3]
        Ex = sp.csr_matrix((data, (row, col)), shape=(self.nx-2, self.nx-2))
        Ax = Ex + Ex.T - 2*sp.identity(self.nx-2)

        # (ny-2) by (ny-2) interior point [1, -2, 1] banded matrix
        data = np.ones(self.ny-3)
        row = np.r_[1:self.ny-2]
        col = np.r_[0:self.ny-3]
        Ey = sp.csr_matrix((data, (row, col)), shape=(self.ny-2, self.ny-2))
        Ay = Ey + Ey.T - 2*sp.identity(self.ny-2)

        # (nx-2)*(ny-2) by (nx-2)*(ny-2) interior point tri-diagonal matrix
        self.coeff_matrix = \
            sp.kron(Ay/self.hy**2, sp.identity(self.nx-2)) + \
            sp.kron(sp.identity(self.ny-2), Ax/self.hx**2)
        self.coeff_built = True

    def build_rhs_vector(self):
        """
        Computes the
        """
        # (nx-2)*(ny-2) source vector for the system
        self.calculate_boundary_terms()
        self.rhs_vector = np.ravel(self.source_terms - self.bc_terms, order='F')
        self.rhs_built = True

    def apply_boundaries(self):
        # Dirichlet BCs:
        if self.dirichlet['north'] is not None:
            self.solution[:, -1] = self.dirichlet['north']

        if self.dirichlet['south'] is not None:
            self.solution[:, 0] = self.dirichlet['south']

        if self.dirichlet['east'] is not None:
            self.solution[-1, :] = self.dirichlet['east']

        if self.dirichlet['west'] is not None:
            self.solution[0, :] = self.dirichlet['west']

        # Neumann BCs:
        if self.neumann['north'] is not None:
            du_north = self.neumann['north']
            self.solution[:, -1] = self.solution[:, -2] + du_north*self.hy

        if self.neumann['south'] is not None:
            du_south = self.neumann['north']
            self.solution[:, 0] = self.solution[:, 1] - du_south*self.hy

        if self.neumann['east'] is not None:
            du_east = self.neumann['east']
            self.solution[-1, :] = self.solution[-2, :] + du_east*self.hx

        if self.neumann['west'] is not None:
            du_west = self.neumann['west']
            self.solution[0, :] = self.solution[1, :] - du_west*self.hx

    def direct_solve(self):
        """
        Perform a system solution using a direct method
        """
        # make sure the system has been prepared for the solution
        if not self.coeff_built:
            self.build_coeff_matrix()
        if not self.rhs_built:
            self.build_rhs_vector()
        # perform the solution
        u = sp.linalg.spsolve(self.coeff_matrix, self.rhs_vector)
        # map the solution onto the domain
        self.solution[1:-1, 1:-1] = np.reshape(u, (self.nx-2, self.ny-2),
                                               order='F')
        # apply boundary conditions
        self.apply_boundaries()
        print('Solution succeeded!')

    def plot_field(self, show=True, save=False):
        """
        Plot solution.
        """
        X = np.linspace(0.0, self.lx, num=self.nx)
        Y = np.linspace(0.0, self.ly, num=self.ny)
        X, Y = np.meshgrid(X, Y)
        fig = plt.figure()
        ax1 = fig.add_subplot(1,1,1,projection='3d')
        surf1 = ax1.plot_surface(
            X, Y, self.solution, rstride=1, cstride=1, cmap=cm.coolwarm,
            linewidth=0, antialiased=False)
        plt.xlabel('X')
        plt.ylabel('Y')
        ax1.set_zlim(0, 1)
        plt.colorbar(surf1, shrink=0.5, aspect=5)
        ax1.view_init(30,45)
        if show:
            plt.show()
        if save:
            plt.savefig('laplace2d.png')
