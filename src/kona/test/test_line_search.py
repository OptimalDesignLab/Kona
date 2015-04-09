import unittest

from kona.algorithms.util.linsearch import BackTracking
from kona.algorithms.util.merit import ObjectiveMerit


class BackTrackingTestCase(unittest.testcase):


    def setUp(self):
        self.bt = BackTracking()

    def test_stops_after_multiple_iter(self):
        '''Assuming your first guess viloates sufficient decrease condition'''
        self.fail("untested")

    def test_stops_after_one_iter(self):
        '''Assuming your first guess viloates sufficient decrease condition'''
        self.fail("untested")

    def test_from_left_to_right(self):
        '''Assuming your first guess viloates sufficient decrease condition'''
        self.fail("untested")

    def test_from_right_to_left(self):
        '''Assuming your first guess viloates sufficient decrease condition'''
        self.fail("untested")
