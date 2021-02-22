"""
Functions related to THOR simulation
"""
import numpy as np
from ai2thor.controller import Controller

def reachable_thor_pos2d(controller):
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
