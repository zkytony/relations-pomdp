import copy

class Object:
    """
    An object is just an id. So, it is EQUIVALENT to
    the id, that is, an integer.

    An object can store attributes.
    """
    def __init__(self, objid, attributes):
        self.id = objid
        self.attributes = attributes

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, int):
            return self.id == other
        elif isinstance(other, Object):
            return self.id == other.id
        else:
            return False

    def __getitem__(self, attr):
        return self.attributes[attr]

    def get(self, attr, default_val):
        return self.attributes.get(attr, default_val)
