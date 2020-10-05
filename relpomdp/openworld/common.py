from abc import ABC
import random

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
    
class Vec2d(Attr):
    """A 2D array"""
    def __init__(self, val):
        assert len(val) == 2, "`val` needs to be of length 2!"
        self._val = val

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


class Domain(ABC):
    @abstractmethod
    def check(self, val, verbose=True):
        """Returns true if value `val` is in Domain.
        Will not raise exception if val is of invalid
        format - will return False in this case.
        If verbose, print a message to explain why."""
        pass
    @abstractmethod
    def sample(self):
        """Returns a value sampled from this domain;
        The underlying distribution could be anything you want."""
        pass
    # Implement below if the domain is enumerable
    @abstractmethod
    def __iter__(self):
        pass
    @abstractmethod
    def __next__(self):
        pass    

    
class Ranges(Domain):
    """
    A convenient type of Domain where the acceptable
    values can be a single value, or array of any
    dimension. Creating a Ranges domain requires
    specifying a scope, which is either a 2-tuple
    or a set for every entry in the value; When
    the scope is a 2-tuple, it is assumed to be
    the min/max values of a continuous range. For
    integer range like 1 to 100, just pass in the set
    of 100 numbers (TODO: improve this?)
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
                return False, "Value {} not in range {}".format(v, rang)
        else:
            if v not in rang:
                return False, "Value {} not in range {}".format(v, rang)
        return True, ""

    def check(self, val, verbose=True):
        res = True
        msg = ""
        if hasattr(val, "len"):
            if len(val) != len(self._ranges):
                msg = "Number of elements in"\
                    "value {} != that declared in Ranges {}"\
                    .format(len(val), len(self._ranges))
                res = False
            else:
                for i, v in enumerate(val):
                    rang = self._ranges[i]
                    res, msg = self._check_in_range(rang, v)
                    if not res:
                        break
        else:
            if len(self._ranges) != 1:
                msg = "Number of elements in"\
                    "value 1 != that declared in Ranges {}"\
                    .format(len(self._ranges))
                res = False
            rang = self._ranges[0]
            res, msg = self._check_in_range(rang, val)
        if verbose and len(msg) > 0:
            print("Warning: ", msg)
        return res
    
    def sample(self, samplers=[], joint_sampler=None):
        """Samples independently and uniformly from the ranges. For non-uniform
        independent sampling, you can pass. in `samplers` which contains a list
        of functions in the order of the `ranges` passed in to this Ranges
        constructor, a function that maps a range to a value.

        For non-uniform, non-independent sampling, you can pass in a 'joint_sampler',
        which is a function that takes in 'ranges' of this object and returns a value.
        (I do not expect this to be used much or at all.)
        """
        if joint_sampler is not None:
            return joint_sampler.sample(self._ranges)
        elems = []
        for i, rang in enumerate(self._ranges):
            if len(samplers) > i:
                elems.append(samplers[i].sample(rang))
            if type(rang) == tuple:
                return random.uniform(*rang)
            else:
                return random.sample(rang, 1)[0]
            
            
        
