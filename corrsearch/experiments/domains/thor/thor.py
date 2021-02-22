"""
Functions related to THOR simulation
"""
import numpy as np
from ai2thor.controller import Controller
from corrsearch.experiments.domains.thor.grid_map import GridMap
from corrsearch.utils import remap

def reachable_thor_loc2d(controller):
    """
    Returns a tuple (x, z) where x and z are lists corresponding to x/z coordinates.
    You can obtain a set of 2d positions tuples by:
        `set(zip(x, z))`
    """
    # get reachable positions
    event = controller.step(action="GetReachablePositions")
    positions = event.metadata["actionReturn"]
    x = np.array([p['x'] for p in positions])
    y = np.array([p['y'] for p in positions])
    z = np.array([p['z'] for p in positions])
    return x, z

def launch_controller(config):
    controller = Controller(scene=config["scene_name"],
                            agentMode=config.get("agent_mode", "bot"),
                            width=config.get("width", 300),
                            height=config.get("height", 300),
                            gridSize=config.get("grid_size", 0.25),
                            renderDepthImage=config.get("render_depth", True),
                            renderClassImage=config.get("render_class", True),
                            renderObjectImage=config.get("render_object", True))
    return controller


def thor_get(controller, *keys):
    """Get the true environment state, which is the metadata returned
    by the controller. If you would like a particular state variable's value,
    pass in a sequence of string keys to retrieve that value.
    For example, to get agent pose, you call:

    env.state("agent", "position")"""
    event = controller.step(action="Pass")
    if len(keys) > 0:
        d = event.metadata
        for k in keys:
            d = d[k]
        return d
    else:
        return event.metadata

def thor_agent_pose2d(controller):
    """Returns a tuple (x, y, th), a 2D pose
    """
    position = thor_get(controller, "agent", "position")
    rotation = thor_get(controller, "agent", "rotation")
    return position["x"], position["z"], rotation["y"]

def thor_agent_pose(controller):
    """Returns a tuple (pos, rot),
    pos: dict (x=, y=, z=)
    rot: dict (x=, y=, z=)
    """
    position = thor_get(controller, "agent", "position")
    rotation = thor_get(controller, "agent", "rotation")
    return position, rotation

def thor_apply_pose(controller, pose):
    """Given a 2d pose (x,y,th), teleport the agent to that pose"""
    pos, rot = thor_agent_pose(controller)
    x, z, th = pose
    # if th != 0.0:
    #     import pdb; pdb.set_trace()
    controller.step("TeleportFull",
                    x=x, y=pos["y"], z=z,
                    rotation=dict(y=th))


def convert_scene_to_grid_map(controller, scene_name, grid_size):
    """Converts an Ai2Thor scene to a GridMap"""
    x, z = reachable_thor_loc2d(controller)

    # obtain grid indices for coordinates  (origin NOT at (0,0))
    thor_gx = (x // grid_size).astype(int)
    thor_gy = (z // grid_size).astype(int)
    width = max(thor_gx) - min(thor_gx)
    length = max(thor_gy) - min(thor_gy)

    # save these for later use
    thor_gx_range = (min(thor_gx), max(thor_gx))
    thor_gy_range = (min(thor_gy), max(thor_gy))

    # remap coordinates to be nonnegative (origin AT (0,0))
    gx = remap(thor_gx, min(thor_gx), max(thor_gx), 0, width).astype(int)
    gy = remap(thor_gy, min(thor_gy), max(thor_gy), 0, length).astype(int)

    # Little test: can convert back
    try:
        assert all(remap(gx, min(gx), max(gx), thor_gx_range[0], thor_gx_range[1]).astype(int) == thor_gx)
        assert all(remap(gy, min(gy), max(gy), thor_gy_range[0], thor_gy_range[1]).astype(int) == thor_gy)
    except AssertionError as ex:
        print("Unable to remap coordinates")
        raise ex

    # grid map positions
    positions = set(zip(gx, gy))

    # grid map dimensions
    # obstacles: locations that do not fall into valid positions
    obstacles = {(x,y)
                 for x in gx
                 for y in gy
                 if (x,y) not in positions}

    grid_map = GridMap(width, length, obstacles,
                       name=scene_name,
                       ranges_in_thor=(thor_gx_range, thor_gy_range))

    return grid_map
