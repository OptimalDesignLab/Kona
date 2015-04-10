# Helpers to make the BadKonaOption work

def make_optn_key_str(keys):
    return "".join(["['%s']"%k for k in keys])

def get_opt(optns, default, *keys):
    '''
    utility function to make it easier to work with
    nested options dictionaries

    Parameters
    ------------------
    optns: nested dict
    default: value to return if no option found

    *keys: string keys for the nested dictionary
    '''

    keys = list(keys)
    k = keys.pop(0)
    val = optns.get(k, default)
    if isinstance(val, dict):
        return get_opt(val, default, *keys)
    return val

class BadKonaOption(Exception):

    def __init__(self, optns, *keys):
        self.val  = get_opt(optns, None, *keys)
        self.keys = keys

    def __str__(self):
        optns_str = make_optn_key_str(self.keys)
        return "Invalid Kona option: optns%s = %s" % (optns_str, self.val)
