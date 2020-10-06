# initialize this domain. Using initializer functionality in openworld
from domain import CLASSES, OBJECTS
from relpomdp.openworld.initializer import build_object_states


print(build_object_states(CLASSES, OBJECTS))
