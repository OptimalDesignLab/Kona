from optimizer import Optimizer
import linalg
import algorithms
import user
import examples


# Helpers to make the BadKonaOption work

def make_optn_key_str(keys):
    return "".join(["['%s']"%k for k in keys])

def nested_dict_get(dictionary, keys):
    k = keys.pop(0)
    val = dictionary[k]
    if isinstance(val, dict):
        return nested_dict_get(val, keys)
    return val


class BadKonaOption(Exception):

    def __init__(self, optns, keys):
        self.val  = nested_dict_get(optns, list(keys))
        self.keys = keys

    def __str__(self):
        optns_str = make_optn_key_str(self.keys)
        return "Invalid Kona option: optns%s = %s" % (optns_str, self.val)
