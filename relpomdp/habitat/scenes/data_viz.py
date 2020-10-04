import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import random
random.seed(105)

import pandas as pd
import numpy as np
import relpomdp.utils as util


def plot_object(ax, center, size, color, catg):
    cx, cy, cz = center
    lx, ly, lz = size

    xlim = (cx-lx/2, cx+lx/2)
    ylim = (cy-ly/2, cy+ly/2)
    zlim = (cz-lz/2, cz+lz/2)

    # x, y, z coordinates of vertices of the box
    corner_x = []
    corner_y = []
    corner_z = []
    for x in xlim:
        for y in ylim:
            for z in zlim:
                corner_x.append(x)
                corner_y.append(y)
                corner_z.append(z)
    ax.scatter(corner_x, corner_y, corner_z, color=color)
    corners = np.array([corner_x, corner_y, corner_z]).transpose()
    # This is a list of the sides (each a 4-side polygon) that make up the box
    # Since each side (i.e. each row in this list) is a 4-side polygon,
    # the shape of `sides` is 8x4x3. This is the expected shape of Poly3DCollection
    sides = np.array([[corners[0],corners[1],corners[3],corners[2]],
                      [corners[4],corners[5],corners[7],corners[6]], 
                      [corners[0],corners[1],corners[5],corners[4]], 
                      [corners[2],corners[3],corners[7],corners[6]], 
                      [corners[1],corners[3],corners[7],corners[5]],
                      [corners[4],corners[6],corners[2],corners[0]]]
    )
    box = Poly3DCollection(sides,
                           facecolor=color,
                           linewidth=0.2,
                           edgecolors='r',
                           alpha=0.25)                           
    ax.add_collection3d(box)

def plot_objects_for_scene(ax, df, scene_name,
                           included_catgs=None,
                           excluded_catgs=None):
    rows = df.loc[df["scene_name"] == scene_name]
    catg2color = {}
    
    for i, row in rows.iterrows():
        x = row["x"]
        z = row["y"]
        y = row["z"]
        lx = row["size_x"]
        lz = row["size_y"]
        ly = row["size_z"]
        catg = row["category"]
        if included_catgs is not None\
           and catg not in included_catgs:
            continue
        if excluded_catgs is not None\
           and catg in excluded_catgs:
            continue
        if catg not in catg2color:
            if len(catg2color) > 20:
                continue
            colors = [catg2color[c] for c in catg2color]
            catg2color[catg] = util.random_unique_color(colors, ctype=1)
        plot_object(ax, (x,y,z), (lx,ly,lz), catg2color[catg], catg)


    lines = []
    catgs = []
    for catg in catg2color:
        fake2Dline = mpl.lines.Line2D([0],[0],
                                      linestyle="none", c=catg2color[catg],
                                      marker='o')
        lines.append(fake2Dline)
        catgs.append(catg)
    ax.legend(lines, catgs, numpoints=1)


scene_name = "frl_apartment_0"
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

df = pd.read_csv("./replica_objects.csv")

included={"bowl", "book", "countertop", "shelf", "wall-cabinet", "cabinet"}
excluded={"wall", "ceiling", "lamp", "window",
          "vent", "door", "blinds", "wall-plug", "picture"}
plot_objects_for_scene(ax, df, scene_name,
                       excluded_catgs=excluded,
                       included_catgs=included)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title(scene_name)
plt.tight_layout()
plt.show()

    

