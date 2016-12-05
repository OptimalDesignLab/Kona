import sys

def make_optn_key_str(keys):
    return "".join(["['%s']"%k for k in keys])

def get_opt(optns, default, *keys):
    """
    Utility function to make it easier to work with nested options dictionaries.

    Parameters
    ----------
    optns : dict
        Nested dictionary.
    default : Unknown
        Value to return of the dictionary is empty.
    \*keys : string
        Keys from which value will be pulled

    Returns
    -------
    Unknown
        Dictionary value corresponding to given hierarchy of keys.
    """
    keys = list(keys)

    k = keys.pop(0)
    val = optns.get(k, default)
    if isinstance(val, dict) and bool(val) and bool(keys):
        return get_opt(val, default, *keys)
    return val

def print_dict(obj, pre='', out_file=sys.stdout):
    for k, v in obj.items():
        if hasattr(v, '__iter__'):
            out_file.write('%s%s : {\n'%(pre, k))
            print_dict(v, pre='%s  '%pre, out_file=out_file)
            out_file.write('%s}\n'%pre)
        else:
            out_file.write('%s%s : %s\n'%(pre, k, v))

class BadKonaOption(Exception):
    """
    Special exception class for identifying bad Kona configuration options.

    Parameters
    ----------
    optns : dict
        Options dictionary containing the bad configuration.
    \*keys : string
        Hierarchy of dictionary keys identifying the bad configuration.
    """
    def __init__(self, optns, *keys):
        self.val  = get_opt(optns, None, *keys)
        self.keys = keys

    def __str__(self):
        optns_str = make_optn_key_str(self.keys)
        return "Invalid Kona option: optns%s = %s" % (optns_str, self.val)
