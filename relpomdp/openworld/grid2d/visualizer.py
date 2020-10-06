from relpomdp.openworld.visualizer import WorldViz
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

class CupboardViz:
    @classmethod
    def update(cls, ax, objstate):
        x, y = objstate["Location"].value
        w, l = objstate["Dimension"].value        
        facing = objstate["Facing"].value
        door_angle = objstate["DoorAngle"].value
        rect = patches.Rectangle((x,y), w, l, facing, color="red")
        ax.add_patch(rect)


class GridWorldViz(WorldViz):
    """Visualize the world using matplotlib"""

    def __init__(self, metric_map):

        self.classviz = {
            "Cupboard": CupboardViz
        }
        self.metric_map = metric_map
        self.fig, self.ax = plt.subplots()
        plt.ion()

        # plot the map;
        xvals = []
        yvals = []
        for x, y in metric_map.obstacle_poses:
            rect = patches.Rectangle((x,y), 1, 1, 0, color="black")
            self.ax.add_patch(rect)
        self.ax.set_xticks(np.arange(metric_map.width), minor=True)
        self.ax.set_yticks(np.arange(metric_map.length), minor=True)
        self.ax.set_xlim(0, metric_map.width)
        self.ax.set_ylim(0, metric_map.length)
        self.ax.grid(which='both')    # Show gridlins for every x/y value
        self.ax.set_aspect("equal")   # make sure the grids are squers
        self.ax.invert_yaxis()   # because when making the map the origin is top-left
    

    def update(self, object_states):
        for objid in object_states:
            objstate = object_states[objid]
            if objstate.objclass in self.classviz:
                self.classviz[objstate.objclass].update(self.ax, objstate)

    def clear(self):
        self.ax.clear()
        self.fig.clf()
