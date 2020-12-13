# Start interactive scene in ai2thor
import argparse
from ai2thor.controller import Controller
import ai2thor_utils as a2til
from pynput import keyboard
import matplotlib.pyplot as plt
import matplotlib.colors as mcl
import cv2
import numpy as np
import time
import re
from pprint import pprint


KEY_TO_ACTION = {
    "d": "MoveAhead",
    "s": "MoveBack",
    "f": ("RotateRight", {'degrees':15.0}),
    "a": ("RotateLeft", {'degrees':15.0}),
    "w": "LookUp",
    "x": "LookDown",
    "p": "GetReachablePositions"
}

class Simulator:
    def __init__(self, controller):
        self.controller = controller
        self.keyboard_listener = keyboard.Listener(on_press=self._on_press)

    def _on_press(self, key):
        if hasattr(key, "char"):
            print("YOU PRESSED {}".format(key.char))
            if key.char in KEY_TO_ACTION:
                action = KEY_TO_ACTION[key.char]
                params = {}
                if type(action) == tuple:
                    action, params = action
                event = self.controller.step(action=action, **params)
                img = event.class_segmentation_frame #depth_frame
                # img /= np.max(img)
                # img *= 255
                print("Visible objects:", a2til.visible_objects(event.metadata))
                print("Action return:", event.metadata["actionReturn"])
                if action == "GetReachablePositions":
                    img = a2til.reachable_locations(event.metadata)
                cv2.imshow('img-depth', img.astype(np.uint8))
                cv2.waitKey(0)

def main():
    parser = argparse.ArgumentParser(
        description="Runs an ai2thor scene with keyboard interaction")
    parser.add_argument("scene_name", help="Scene name. E.g. FloorPlan30, FloorPlan_Train1_3")
    parser.add_argument("--agent-mode", help="agentMode (default, bot or drone)", default="default")
    parser.add_argument("-W", "--width", help="image width", default=300, type=int)
    parser.add_argument("-H", "--height", help="image width", default=300, type=int)
    args = parser.parse_args()

    scene_name = args.scene_name

    # ^...$ means full match
    if re.search("^[0-9]+_[0-9]+$", scene_name) is not None:
        scene_name = "FloorPlan_Train%s" % scene_name

    controller = Controller(scene=scene_name,
                            agentMode=args.agent_mode,
                            width=args.width,
                            height=args.height,
                            renderDepthImage=True,
                            renderClassImage=True,
                            renderObjectImage=True)
    event = controller.step(action="Pass")
    pprint(a2til.scene_info(event.metadata))

    simulator = Simulator(controller)
    simulator.keyboard_listener.start()
    simulator.keyboard_listener.join()

if __name__ == "__main__":
    main()
