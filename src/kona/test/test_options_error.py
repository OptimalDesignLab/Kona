import unittest

from kona.options import BadKonaOption

class KonaOptionTestCase(unittest.TestCase):

    def test_err_msg(self):
        optns = {
            'quasi_newton': {'type': 2}
        }
        try:
            raise BadKonaOption(optns, 'quasi_newton', 'type')
        except BadKonaOption as err:
            self.assertEqual(
                str(err),
                "Invalid Kona option: optns['quasi_newton']['type'] = 2")
        else:
            self.fail('BadKonaOption expected')


if __name__ == "__main__":
    unittest.main()
