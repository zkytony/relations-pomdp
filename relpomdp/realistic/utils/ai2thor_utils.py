# Utility functions for ai2thor
import matplotlib.pyplot as plt
import matplotlib.colors as mcl
import cv2
import os
import numpy as np
import math
from moos3d.util_viz import plot_voxels, CMAPS
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


def visible_objects(metadata):
    result = []
    for obj in metadata["objects"]:
        if obj["visible"]:
            result.append(obj["name"])
    return result

def scene_info(metadata):
    typecount = {}
    for obj in metadata["objects"]:
        if obj["objectType"] not in typecount:
            typecount[obj["objectType"]] = 0
        typecount[obj["objectType"]] += 1
    return {"TypeCount": typecount}

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

def save_frames(controller, actions, savepath,
                prefix="frame", frame_type="rgb", step_cb=None, step_cb_args={}):

    """Pass in a controller, and a sequence of actions.
    Execute these actions, save the frames as images.

    Args:
        actions (iterable): sequence of tuples, (action_name, params)

    Note that this alters the world state, so cannot be reset."""
    i = 0
    os.makedirs(savepath, exist_ok=True)
    for action_name, params in actions:
        event = controller.step(action=action_name, **params)
        if step_cb is not None:
            step_cb(i, event, **step_cb_args)

        if frame_type == "rgb":
            img = event.frame
        elif frame_type == "depth":
            img = even.depth_frame
        elif frame_type == "class":
            img = even.class_segmentation_frame
        elif frame_type == "object":
            img = even.instance_segmentation_frame
        else:
            raise ValueError("Unknown frame_type:", frame_type)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(savepath, "%s-%d.png" % (prefix, i)), img)
        i += 1


def plot_reachable_grid(controller, ax, agent_pose=None):
    """Creates a scatter plot of the reachable positions
    and plots the agent position on top if given."""
    event = controller.step(action="GetReachablePositions")
    positions = event.metadata["actionReturn"]

    ax.clear()
    x = np.array([p['x'] for p in positions])
    y = np.array([p['y'] for p in positions])
    z = np.array([p['z'] for p in positions])
    # Plots the boundaries floor
    ax.scatter(x, z, c='r')
    ax.scatter([min(x)]*len(positions), z, c='y')

    if agent_pose is not None:
        pos, rot = agent_pose
        ax.scatter([pos[0]], [pos[2]], c='b')
        # head = (pos[0] + math.sin(rot[1])*0.25,
        #         pos[2] + math.cos(rot[1])*0.25)
        # ax.plot([pos[0], head[0]], [pos[2], head[1]], 'bo-')
        print(pos, rot)


def get_reachable_pos_set(controller, use_2d=False):
    event = controller.step(action="GetReachablePositions")
    positions = event.metadata["actionReturn"]
    x = np.array([p['x'] for p in positions])
    y = np.array([p['y'] for p in positions])
    z = np.array([p['z'] for p in positions])
    if use_2d:
        return set(zip(x, z))
    else:
        return set(zip(x, y, z))
