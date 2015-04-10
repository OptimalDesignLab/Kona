# Helpers to make the BadKonaOption work

def make_optn_key_str(keys):
    return "".join(["['%s']"%k for k in keys])

def get_opt(dictionary, *keys):
    keys = list(keys)
    k = keys.pop(0)
    val = dictionary.get(k, None)
    if isinstance(val, dict):
        return get_opt(val, *keys)

    return val

class BadKonaOption(Exception):

    def __init__(self, optns, *keys):
        self.val  = get_opt(optns, *keys)
        self.keys = keys

    def __str__(self):
        optns_str = make_optn_key_str(self.keys)
        return "Invalid Kona option: optns%s = %s" % (optns_str, self.val)
