"""
Build a topo map for a THOR env
"""
from corrsearch.experiments.domains.thor.thor import *
from corrsearch.experiments.domains.thor.visualizer import *
from corrsearch.experiments.domains.thor.tests import DummyProblem
import cv2
import time
import pygame
from pprint import pprint

def get_clicked_pos(r, w, l):
    pos = None
    while pos is None:
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = event.pos
                break
        time.sleep(0.001)
    # y coordinate is flipped
    return pos[0] // r, l - pos[1] // r - 1


def mark_cell(img, pos, nid, r, linewidth=1, unmark=False):
    if unmark:
        color = (255, 255, 255, 255)
    else:
        color = (53, 190, 232, 255)
    x, y = pos
    cv2.rectangle(img, (y*r, x*r), (y*r+r, x*r+r),
                  color, -1)
    # Draw boundary
    cv2.rectangle(img, (y*r, x*r), (y*r+r, x*r+r),
                  (0, 0, 0), linewidth)

    if not unmark:
        font                   = cv2.FONT_HERSHEY_COMPLEX_SMALL
        fontScale              = 1
        fontColor              = (240, 140, 130)
        lineType               = 2
        imgtxt = np.full((r, r, 4), color, dtype=np.uint8)
        text_loc = (r//4, int(round(r/1.5)))
        cv2.putText(imgtxt, str(nid), text_loc, #(y*r+r//4, x*r+r//2),
                    font, fontScale, fontColor, lineType)
        imgtxt = cv2.rotate(imgtxt, cv2.ROTATE_90_CLOCKWISE)
        img[x*r:x*r+r, y*r:y*r+r] = imgtxt
    return img

def builder(scene_name, grid_size=0.25):
    config = {
        "scene_name": scene_name,
        "width": 400,
        "height": 400,
        "grid_size": 0.25
    }
    robot_id = 0
    controller = launch_controller(config)
    scene_info = load_scene_info(scene_name, data_path="../data")
    grid_map = convert_scene_to_grid_map(controller, scene_info, grid_size)

    # Display grid map, starts interaction
    problem = DummyProblem(robot_id)
    problem.grid_map = grid_map

    viz = ThorViz(problem)
    img = viz.gridworld_img()
    viz.show_img(img)

    topo_spec = {"nodes": {}, "edges": []}

    OPTIONS = {
        0: "add_node",
        1: "add_edge",
        2: "remove_node",
        3: "remove_edge",
        4: "save",
        5: "exit"
    }
    while True:
        pprint(OPTIONS)
        opt = int(input("Action [{}-{}]: ".format(min(OPTIONS), max(OPTIONS))))
        action = OPTIONS[opt]

        if action == "add_node":
            print("Click on window to select cell")
            x, y = get_clicked_pos(viz._res, grid_map.width, grid_map.length)
            print(x, y)

            # while True:
            #     x = int(input("Coordinate x: "))
            #     y = int(input("Coordinate y: "))
            #     if (x,y) not in grid_map.free_locations:
            #         print("{} is not at reachable pose; Re-enter.".format((x,y)))
            #     else:
            #         break
            nid = len(topo_spec["nodes"])
            img = mark_cell(img, (x,y), nid, viz._res, viz._linewidth)
            thor_pos = grid_map.to_thor_pos(x, y, grid_size=config["grid_size"])
            topo_spec["nodes"][str(nid)] = {"x": thor_pos[0], "z": thor_pos[1]}
            viz.show_img(img)
            print("Added node {} to {} (aka. {})".format(nid, thor_pos, (x,y)))


if __name__ == "__main__":
    builder("FloorPlan_Train1_1", grid_size=0.25)
