[![Build Status](https://travis-ci.org/OptimalDesignLab/Kona.svg?branch=master)](https://travis-ci.org/OptimalDesignLab/Kona)
[![Coverage Status](https://coveralls.io/repos/OptimalDesignLab/Kona/badge.svg?branch=master)](https://coveralls.io/r/OptimalDesignLab/Kona?branch=master)
[![Documentation Status](https://readthedocs.org/projects/kona/badge/?version=latest)](http://kona.readthedocs.org/en/latest/)

# Kona - A Parallel Optimization Framework

Kona is a library for nonlinear constrained optimization. It was designed
primarily for large-scale partial-differential-equation (PDE) governed
optimization problems; however it is suitable for any (sufficiently smooth)
problem where the objective function and/or constraints require the solution of
a computational expensive state equation. Kona is also useful for developing
new optimization algorithms for PDE-governed optimization, as a consequence of
its abstracted vector and matrix implementations.

Please refer to the code documentation, API reference and use examples on [ReadTheDocs](http://kona.readthedocs.org) for more details on Kona's
parallel-agnostic implementation and linear algebra abstraction.

An older version of Kona originally written in C++ can be found
[on BitBucket](https://bitbucket.org/odl/kona). This version is also operable
through Python, but is now deprecated in favor of the complete Python re-write.
