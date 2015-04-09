from distutils.core import Command
from setuptools import setup, find_packages

def discover_and_run_tests():
    import os
    import sys
    import unittest

    # get setup.py directory
    setup_file = sys.modules['__main__'].__file__
    setup_dir = os.path.abspath(os.path.dirname(setup_file))

    # use the default shared TestLoader instance
    test_loader = unittest.defaultTestLoader

    # use the basic test runner that outputs to sys.stderr
    test_runner = unittest.TextTestRunner()

    # automatically discover all tests
    # NOTE: only works for python 2.7 and later
    test_suite = test_loader.discover(setup_dir)

    # run the test suite
    test_runner.run(test_suite)

class DiscoverTest(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        discover_and_run_tests()

setup(name = 'kona',
      version = '1.0',
      author = 'Jason E. Hicken',
      author_email = 'hickej2@rpi.edu',
      url = 'https://github.com/OptimalDesignLab/Kona',
      package_dir = {'':'src'},
      packages = find_packages(),
      cmdclass = {'test': DiscoverTest},
      install_requires=[
        'sphinx>=1.3.1',
        'numpydoc>=0.5',
        'numpy>1.9', 
      ],
      )
