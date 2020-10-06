from abc import ABC, abstractmethod
import random

### Attribute
class Attr(ABC):
    """
    Attribute; Can be constructed with
    a single argument "value".
    """
    @property
    def value(self):
        return self._val

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        if isinstance(other, Attr):
            return self.value == other.value
        return False
    
    def __str__(self):
        return "{}({})".format(self.__class__.__name__, self.value)
    
    def __repr__(self):
        return str(self)    
    
    
class Vec2d(Attr):
    """A 2-element array"""
    def __init__(self, val):
        assert len(val) == 2, "`val` needs to be of length 2!"
        self._val = val
    def __iter__(self):
        return iter(self.value)
    def __len__(self):
        return 2

class Real(Attr):
    """A float number"""
    def __init__(self, val):
        self._val = float(val)

class Catg(Attr):
    """A string value"""
    def __init__(self, val):
        assert type(val) == str, "Catg only accepts string."
        self._val = val

class Bool(Attr):
    """A boolean"""
    def __init__(self, val):
        if type(val) == int:
            val = bool(val)
        else:
            assert type(val) == bool
        self._val = val

### Domain
class Domain(ABC):
    @abstractmethod
    def check(self, val):
        """Returns true if value `val` is in Domain.
        Will not raise exception if val is of invalid
        format - will return False in this case.
        If verbose, print a message to explain why.
        Note that val is assumed to NOT be a Attr object,
        but a value (like a float)."""
        pass
    @abstractmethod
    def sample(self, attr_class):
        """Returns a value sampled from this domain; The underlying distribution could
        be anything you want.  Note that this function is computing the value of
        an attribute `attr_class`.
        """
        pass
    # Implement below if the domain is enumerable
    def __iter__(self):
        pass
    def __next__(self):
        pass    

    
class Ranges(Domain):
    """A convenient type of Domain where the acceptable values can be a single
    value, or array of any dimension. Creating a Ranges domain requires
    specifying a scope, which is either a 2-tuple or a set for every entry in
    the value; When the scope is a 2-tuple, it is assumed to be the min/max
    values of a continuous range. For integer range like 1 to 100, just pass in
    the set of 100 numbers (TODO: improve this?)

    Note that if the value is an array, each element has a range and is
    independent from other elements (i.e. sampling from this domain will be
    uniform and independent for each element)
    """
    def __init__(self, *ranges):
        # Check if the ranges is valid
        for r in ranges:
            if not ((type(r) == tuple and len(r) == 2)\
                    or type(r) == set):
                raise ValueError("Invalid range (Must be a 2-tuple or set): {}"\
                                 .format(r))
        self._ranges = list(ranges)
        
    def _check_in_range(self, rang, v):
        if type(rang) == tuple:
            if not (rang[0] <= v <= rang[1]):
                return False
        else:
            if v not in rang:
                return False
        return True#, ""

    def check(self, val):
        res = True
        # msg = ""
        if type(val) == tuple:
            if len(val) != len(self._ranges):
                # Number of elements in vals != that declared in Ranges                
                res = False
            else:
                for i, v in enumerate(val):
                    rang = self._ranges[i]
                    res = self._check_in_range(rang, v)
                    if not res:
                        break
        else:
            if len(self._ranges) != 1:
                # Number of elements in vals != that declared in Ranges
                res = False
            rang = self._ranges[0]
            res = self._check_in_range(rang, val)
        return res
    
    def sample(self, attr_class):
        """Samples independently and uniformly from the ranges,
        as defined above.

        Note that this function is not returning the Attribute, but the
        value. The user is expected to take this value and construct a
        corresponding Attribute object, if necessary. 
        """
        elems = []
        for i, rang in enumerate(self._ranges):
            if type(rang) == tuple:
                elems.append(random.uniform(*rang))
            else:
                elems.append(random.sample(rang, 1)[0])
        if len(elems) == 1:
            return attr_class(elems[0])
        else:
            return attr_class(tuple(elems))

    def __str__(self):
        return "Ranges({})".format(self._ranges)

    def __repr__(self):
        return str(self)

### JOKE
