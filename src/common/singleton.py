"""singleton.py: Contains the Singleton class definition."""

__author__      = "Alp Dener"
__email__       = "denera@rpi.edu"
__license__     = "LGPL"
__date__        = "April 2, 2015"

class Singleton(object):
    """
    This is a non-thread-safe singleton implementation in Python that allows 
    for inheritance. The motivation is that the inherited class will look 
    exactly the same as any other Python class, but behave as a Singleton. Any 
    attempts to initialize the object more than once will raise a Runtime Error.
    """
    
    __instance = None # one instance to rule them all

    def __new__(cls, *args, **kwargs):
        # if __instance is None, this will call the __init__ for the object
        if not cls.__instance:
            cls.__instance = super(Singleton, cls).__new__(cls, *args, **kwargs)
            return cls.__instance
        # otherwise, raise error because we already have an instance created
        else:
            raise RuntimeError('Attempting to reallocate a Singleton!')