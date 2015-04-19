[![Build Status](https://travis-ci.org/OptimalDesignLab/Kona.svg?branch=master)](https://travis-ci.org/OptimalDesignLab/Kona)
[![Coverage Status](https://coveralls.io/repos/OptimalDesignLab/Kona/badge.svg?branch=master)](https://coveralls.io/r/OptimalDesignLab/Kona?branch=master)
[![Documentation Status](https://readthedocs.org/projects/kona/badge/?version=latest)](http://kona.readthedocs.org/en/latest/)

# Kona

## What is it?

Kona is a library for nonlinear constrained optimization.

## Who is it for?

Kona was designed primarily for large-scale partial-differential-equation (PDE)
constrained optimization problems; however, it is suitable for any (sufficiently
smooth) problem where the objective function and/or constraints require the
solution of a computationally expensive state equation.  Kona is also useful for
developing new optimization algorithms for PDE-constrained optimization, for the
reasons described below.

## How is it implemented?

An important aspect of Kona's implementation is that it makes no assumptions
regarding the dimension, type, or parallelization of the primal variables, state
variables, or constraints.  In other words, these objects are assumed to exist in
an abstract vector space.

For ease of use, Kona provides a default
[NumPy](http://dx.doi.org/10.1109/MCSE.2011.37) implementation for these vector
spaces.  For high-performance applications, where variables are typically
distributed across multiple processes, the user must implement the storage and
linear algebra operations for the vector spaces.  This model allows Kona to be
used in a variety of parallel environments, because it remains agnostic to how
the user defines, stores, and manipulates the vectors.

Additionally, Kona separates optimization algorithms from the underlying PDE
solver such that any new optimization algorithm can be implemented in Kona
using Kona's own abstracted vector classes. This allows for rapid development
and testing of new algorithms independently from the PDE solver, and
guarantees that any solver that has already been integrated with Kona will
work correctly underneath any new algorithm that may be added in the future.

## History and References

An older version of Kona written in C++ can be found
[here](https://bitbucket.org/odl/kona).

Insert AIAA paper references here.
