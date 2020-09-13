from relpomdp.oopomdp.framework import ObjectState, Attribute

class AbstractAttribute(Attribute):
    def __init__(self, name, value):
        super().__init__(name, value)

    def reverse_image(self, *args, **kwargs):
        """Returns the set of attributes that map
        to this abstract attribute"""
        raise NotImplementedError

    @classmethod
    def abstractify(self, attribute, *args, **kwargs):
        """Returns an AbstractAttribute
        given another attribute"""
        raise NotImplementedError
