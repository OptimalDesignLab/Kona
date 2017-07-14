[![Build Status](https://travis-ci.org/OptimalDesignLab/Kona.svg?branch=master)](https://travis-ci.org/OptimalDesignLab/Kona)
[![Coverage Status](https://coveralls.io/repos/OptimalDesignLab/Kona/badge.svg?branch=master&service=github)](https://coveralls.io/github/OptimalDesignLab/Kona?branch=master)
[![codecov.io](http://codecov.io/github/OptimalDesignLab/Kona/coverage.svg?branch=master)](http://codecov.io/github/OptimalDesignLab/Kona?branch=master)
[![Documentation Status](https://readthedocs.org/projects/kona/badge/?version=latest)](http://kona.readthedocs.io/)

# Kona - A Parallel Optimization Framework

## Install Guide

If you already have a Python distribution, Kona can be installed using `pip install -e . --user` 
run from the root directory of the repo. 

On macOS, we strongly recommend installing [Miniconda](https://conda.io/miniconda.html) first 
before installing Kona on top. Using the system Python distribution on macOS can break the 
operating system.

For a quick start guide on how to run optimization problems with Kona, or for more detail on 
macOS installations, please refer to the [documentation](http://kona.readthedocs.org).

## What is it?

Kona is a library for nonlinear (equality) constrained optimization. The stable master branch 
currently provides several gradient-based optimization algorithms for both constrained and 
unconstrained problems. Development is ongoing to support inequality constrained problems at a 
future date.

## Who is it for?

Kona was designed primarily for large-scale partial-differential-equation (PDE)
governed optimization problems; however it is suitable for any
(sufficiently smooth) problem where the objective function and/or constraints
require the solution of a computational expensive state equation.

As a consequence of its abstracted vector and matrix implementations, Kona is
also useful for developing new optimization algorithms for PDE-governed
optimization.

Please refer to the API reference and use examples in the
[documentation](http://kona.readthedocs.org) for more details on Kona's
parallel-agnostic implementation and linear algebra abstraction.

## Who is developing it?

Kona was originally written by Dr. Jason E. Hicken in C++. This old version is
now deprecated, but still available [on BitBucket](https://bitbucket.org/odl/kona).

The current Python implementation of Kona is being developed and maintained by the 
[Optimal Design Lab](http://www.optimaldesignlab.com) research group at Rensselaer Polytechnic 
Institute.

## Acknowledgements

This work is supported by the National Science Foundation under Grant No.
1332819, and the National Aeronautics and Space Administration under Grant No. 
NNX14AC73A.
