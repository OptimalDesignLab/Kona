import unittest
import os, sys
from kona.common import Singleton

# define a new class as a Singleton
class Cat(Singleton):
    def __init__(self, name):
        self.name = name

# unit tests begin here
class testSingleton(unittest.TestCase):
        
    def testConstruct(self):
        # construct an instance of the singleton
        pet = Cat('Maple')
        self.assertTrue(isinstance(pet, Cat) and isinstance(pet, Singleton))
        
    def testInstance(self):
        # test if Singleton raises the correct error when new instance is called
        with self.assertRaises(RuntimeError):
            newPet = Cat('Midna')

if __name__ == '__main__':
    unittest.main()