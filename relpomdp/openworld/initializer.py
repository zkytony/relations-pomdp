# Initialize the world from given CLASS and OBJECT specifications
from relpomdp.oopomdp.framework import ObjectState

def build_object_states(classes, objects):
    """
    Given `classes`, a dictionary mapping from class_name,
        to {attribute_name -> (ValueType, Domain)},
    and `objects`, a dictionary mapping from object id,
        to ("object class", (attribute_name, value) ...)
    return a dictionary mapping from object id to ObjectState
    """
    objstates = {}
    for objid in objects:
        objclass = objects[objid][0]
        if objclass not in classes:
            raise ValueError("Class %s is not a defined class" % objclass)

        objattrs = {}
        for attr, val in objects[objid][1:]:
            if attr not in classes[objclass]:
                raise ValueError("Attribute %s is not a defined attribute for %s"\
                            % (attr, objattr))
            valclass, domain = classes[objclass][attr]            
            if not domain.check(val):
                raise ValueError("Value {} is not in domain {}".format(val, domain))
            attrval = valclass(val)
            objattrs[attr] = attrval
        objstates[objid] = ObjectState(objclass, objattrs)
    return objstates


