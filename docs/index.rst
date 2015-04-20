.. KONA documentation master file, created by
   sphinx-quickstart on Mon Apr  6 10:08:35 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to KONA's documentation!
================================

|build| |coverage|

.. |build| image:: https://travis-ci.org/OptimalDesignLab/Kona.svg?branch=master
    :target: https://travis-ci.org/OptimalDesignLab/Kona
    :alt: Build Status

.. |coverage| image:: https://coveralls.io/repos/OptimalDesignLab/Kona/badge.svg?branch=master
    :target: https://coveralls.io/r/OptimalDesignLab/Kona?branch=master
    :alt: Test Coverage

Kona is a library for nonlinear constrained optimization. It was designed
primarily for large-scale partial-differential-equation (PDE) governed
optimization problems; however it is suitable for any (sufficiently smooth)
problem where the objective function and/or constraints require the solution of
a computational expensive state equation. Kona is also useful for developing
new optimization algorithms for PDE-governed optimization as a result of its
abstracted vector and matrix implementations.

Contents:

.. toctree::
    :titlesonly:

    implementation
    references
    api/kona

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
