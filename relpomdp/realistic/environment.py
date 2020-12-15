"""

Tips about Ai2Thor environments:

* Positions are in meters.
* By default: the environment is a grid world with 'gridSize'. Therefore,
  all navigation actions are discrete, on top of a grid. The getReachablePositions
  obtains all positions that can be reached.
"""


import pomdp_py
from ai2thor.controller import Controller

class ThorEnv:

    def __init__(self, config):
        self.config = config
        self.controller = None

    def launch(self):
        controller = Controller(scene=self.config["scene_name"],
                                agentMode=self.config.get("agent_mode", "default"),
                                width=self.config.get("width", 300),
                                height=self.config.get("height", 300),
                                gridSize=self.config.get("grid_size", 0.25),
                                renderDepthImage=self.config.get("render_depth", True),
                                renderClassImage=self.config.get("render_class", True),
                                renderObjectImage=self.config.get("render_object", True))
        self.controller = controller

    def get(self, *keys):
        """Get the true environment state, which is the metadata returned
        by the controller. If you would like a particular state variable's value,
        pass in a sequence of string keys to retrieve that value.
        For example, to get agent pose, you call:

        env.state("agent", "position")"""
        event = self.controller.step(action="Pass")
        if len(keys) > 0:
            d = event.metadata
            for k in keys:
                d = d[k]
            return d
        else:
            return event.metadata

    def agent_pose(self, use_2d=False):
        """Returns a tuple (pos, rot),
        where
        pos = (x, y, z) are the agent position (3D), and
        rot = (rx, ry, rz) are the rotation (3D)
        """
        position = self.get("agent", "position")
        rotation = self.get("agent", "rotation")
        if use_2d:
            pos = (position["x"], position["z"])
            rot = rotation["y"]
        else:
            pos = (position["x"], position["y"], position["z"])
            rot = (rotation["x"], rotation["y"], rotation["z"])
        return pos, rot


# Test
def test():
    config = {
        "scene_name": "FloorPlan30",
        "agent_mode": "default",
        "width": 400,
        "height": 400
    }
    env = ThorEnv(config)
    env.launch()

    print(env.agent_pose())



if __name__ == "__main__":
    test()
