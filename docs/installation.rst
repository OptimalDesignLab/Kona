Quick Start Guide
=================

Kona can be installed just like any other Python module, using the standard
:code:`setup.py` syntax.

.. code::

    python setup.py install

Below is a simple example script that performs gradient-based optimization on
a multidimensional Rosenbrock problem using the L-BFGS algorithm.

.. code:: python

    import kona

    # initialize the problem with the design space size
    num_design = 2
    solver = kona.examples.Rosenbrock(2)

    # get the optimization algorithm handle -- do not initialize
    algorithm = kona.algorithms.ReducedSpaceNewtonKrylov

    # options dictionary -- we only need convergence tolerance for now
    optns = {
        'primal_tol' : 1e-12,
    }

    # initialize the optimization controller
    optimizer = kona.Optimizer(solver, algorithm, optns)

    # run the optimization
    optimizer.solve()

    # print solution
    print solver.curr_design
