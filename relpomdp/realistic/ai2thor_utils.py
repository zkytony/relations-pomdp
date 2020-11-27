# Utility functions for ai2thor
import matplotlib.pyplot as plt
import matplotlib.colors as mcl
import cv2
import numpy as np
from moos3d.util_viz import plot_voxels, CMAPS
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


def visible_objects(metadata):
    result = []
    for obj in metadata["objects"]:
        if obj["visible"]:
            result.append(obj["name"])
    return result

def reachable_locations(metadata):
    # Plot this in 3du
    fig = Figure()
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(1,1,1,projection="3d")

    vp = []
    vr = []
    vc = []
    for loc in metadata["actionReturn"]:
        vp.append([loc["x"], loc["z"], loc["y"]])
        vr.append(0.5)
        vc.append(mcl.to_hex([1.0, 0.0, 0.47, 0.5], keep_alpha=True))

    # import pdb; pdb.set_trace()
    pc = plot_voxels(vp, vr, vc, ax=ax, edgecolor="black", linewidth=0.1)

    vmin = np.min(vp)
    vmax = np.max(vp)
    ax.set_xlim([vmin-0.5, vmax+0.5])
    ax.set_ylim([vmin-0.5, vmax+0.5])
    ax.set_zlim([vmin-0.5, vmax+0.5])
    ax.set_xlabel('X axis')
    ax.set_ylabel('Z axis')
    ax.set_zlabel('Y axis')

    canvas.draw()  # Obtain matplotlib plot content as an image
    width, height = (fig.get_size_inches() * fig.get_dpi()).astype('int')
    image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
    return image
