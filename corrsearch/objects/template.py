import copy

class Object:
    """
    An object is just an id. So, it is EQUIVALENT to
    the id, that is, an integer
    """
    def __init__(self, objid):
        self.id = objid

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, int):
            return self.id == other
        elif isinstance(other, Object):
            return self.id == other.id
        else:
            return False
