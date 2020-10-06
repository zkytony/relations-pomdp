# initialize this domain. Using initializer functionality in openworld
from domain import CLASSES, OBJECTS, MAP
from relpomdp.openworld.initializer import build_object_states
from relpomdp.openworld.grid2d.visualizer import GridWorldViz
import matplotlib.pyplot as plt


objstates = build_object_states(CLASSES, OBJECTS)
viz = GridWorldViz(MAP)
viz.update(objstates)
plt.show()
input("Press ENTER to exit")
