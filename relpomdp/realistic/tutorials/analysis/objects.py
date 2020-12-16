# Plot out the map in a thor env

from ai2thor.controller import Controller
from relpomdp.realistic.environment import ThorEnv
from relpomdp.realistic.utils.ai2thor_utils import save_frames,\
    plot_reachable_grid, get_reachable_pos_set, scene_info, visible_objects

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import pickle


def test():
    config = {
        "scene_name": "FloorPlan_Train1_1",
        "agent_mode": "default",
        "width": 400,
        "height": 400
    }
    env = ThorEnv(config)
    env.launch()

    event = env.controller.step(action="Pass")
    import pdb; pdb.set_trace()
    # positions = event.metadata["actionReturn"]

    # x = np.array([p['x'] for p in positions])
    # y = np.array([p['y'] for p in positions])
    # z = np.array([p['z'] for p in positions])

    # print("unique # of x values: ", len(np.unique(x)))
    # print("unique # of y values: ", len(np.unique(y)))
    # print("unique # of z values: ", len(np.unique(z)))

    # # Plots the boundaries floor
    # fig = plt.figure()
    # ax = fig.add_subplot(1,1,1, projection="3d")
    # ax.scatter(x, z, [min(y)]*len(positions), c='r')
    # ax.scatter([min(x)]*len(positions), z, y, c='y')
    # plt.show()

if __name__ == "__main__":
    test()
