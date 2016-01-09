[![Build Status](https://travis-ci.org/OptimalDesignLab/Kona.svg?branch=master)](https://travis-ci.org/OptimalDesignLab/Kona)
[![Coverage Status](https://coveralls.io/repos/OptimalDesignLab/Kona/badge.svg?branch=master&service=github)](https://coveralls.io/github/OptimalDesignLab/Kona?branch=master)
[![codecov.io](http://codecov.io/github/OptimalDesignLab/Kona/coverage.svg?branch=master)](http://codecov.io/github/OptimalDesignLab/Kona?branch=master)
[![Documentation Status](https://readthedocs.org/projects/kona/badge/?version=latest)](http://kona.readthedocs.org/en/latest/)

# Kona - A Parallel Optimization Framework

## Install Guide

Kona should be installed using `pip install -e .` run from the root directory
of the repo.

For a quick start guide on how to run optimization problems with Kona, please
refer to the [documentation](http://kona.readthedocs.org).

## What is it?

Kona is a library for nonlinear constrained optimization.

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

The Python re-write of Kona is being developed and maintained by Dr. Hicken and
the [Optimal Design Lab](http://www.optimaldesignlab.com) research group at
Rensselaer Polytechnic Institute.
