Quick Start Guide
=================

Kona can be installed just like any other Python module.

.. code::

    pip install -e .

Below is a simple example script that performs gradient-based optimization on
a multidimensional Rosenbrock problem using the reduced-space Newton-Krylov
(RSNK) algorithm.

.. code:: python

    import kona

    # initialize the problem with the design space size
    num_design = 2
    solver = kona.examples.Rosenbrock(2)

    # get the optimization algorithm handle -- do not initialize
    algorithm = kona.algorithms.ReducedSpaceNewtonKrylov

    # options dictionary -- we only need convergence tolerance for now
    optns = {
        'opt_tol' : 1e-12,
    }

    # initialize the optimization controller
    optimizer = kona.Optimizer(solver, algorithm, optns)

    # run the optimization
    optimizer.solve()

    # print solution
    print solver.curr_design

The above optimization run will produce a ``kona_hist.dat`` file tracking
convergence norms across non-linear iterations.
