Implementation Overview
=======================

An important aspect of Kona's implementation is that it makes no assumptions
regarding the dimension, type, or parallelization of the primal variables, state
variables, or constraints. In other words, these objects are assumed to exist in
an abstract vector space.

For ease of use, Kona provides a default
`NumPy <http://dx.doi.org/10.1109/MCSE.2011.37>`_ implementation for these
vector spaces. For high-performance applications, where variables are typically
distributed across multiple processes, the user must implement the storage and
linear algebra operations for the vector spaces. This model allows Kona to be
used in a variety of parallel environments, because it remains agnostic to how
the user defines, stores, and manipulates the vectors.

Additionally, Kona separates optimization algorithms from the underlying PDE
solver such that any new optimization algorithm can be implemented in Kona
using Kona's own abstracted vector classes. This allows for rapid development
and testing of new algorithms independently from the PDE solver, and
guarantees that any solver that has already been integrated with Kona will
work correctly underneath any new algorithm that may be added in the future.
