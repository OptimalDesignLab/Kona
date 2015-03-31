import unittest

# Here's our "unit tests".
class placeholder(unittest.TestCase):

    def autoPass(self):
        self.failUnless(True)

def main():
    unittest.main()

if __name__ == '__main__':
    main()