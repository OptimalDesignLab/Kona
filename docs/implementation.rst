Implementation Overview
=======================

An important aspect of Kona's implementation is that it makes no assumptions
regarding the dimension, type, or parallelization of the state variables. In 
other words, state variables are assumed to exist in an abstract vector space.

For ease of use, Kona provides a default
`NumPy <http://dx.doi.org/10.1109/MCSE.2011.37>`_ implementation for this
vector space. For high-performance applications, where variables are typically
distributed across multiple processes, the user must implement the storage and
linear algebra operations for the vector space. This model allows Kona to be
used in a variety of parallel environments, because it remains agnostic to how
the user defines, stores, and manipulates the vectors.

However this abstraction only exists for the state-space. For design variables 
and constraints, it is assumed that the vector spaces never become too large for 
a single process. Consequently, design and constraint vector spaces are locked 
into the default NumPy implementation. This allows Kona to exploit various 
explicit-algebra tools that significantly impove the functionality and efficiency 
of its optimization algorithms.

Additionally, Kona separates optimization algorithms from the underlying PDE
solver such that any new optimization algorithm can be implemented in Kona
using Kona's own abstracted vector classes. This allows for rapid development
and testing of new algorithms independently from the PDE solver, and
guarantees that any solver that has already been integrated with Kona will
work correctly underneath any new algorithm that may be added in the future.
