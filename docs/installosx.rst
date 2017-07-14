Installation - macOS
====================

In order to maintain system integrity on macOS, the native Python distribution 
should **not** be used for development.

Instead, it is highly recommended that you install [Miniconda](https://conda.io/miniconda.html) 
into your home directory and use this new Python distribution to operate and develop with
Kona.

To install Miniconda, first download [the official installation script](https://repo.continuum.io/miniconda/Miniconda2-latest-MacOSX-x86_64.sh).
Then run:

.. code::

    bash Miniconda2-latest-MacOSX-x86_64.sh

Proceed through the dialogue to install Miniconda's Python 2.7 distribution to your home directory. 
This will also update your `PATH` to make available the newly installed Python executables.

Once installed, grab essential packages using Miniconda's own package manager:

.. code::

    conda install numpy scipy nose pip

Finally we can install Kona just like any other Python module:

.. code::

    pip install -e .