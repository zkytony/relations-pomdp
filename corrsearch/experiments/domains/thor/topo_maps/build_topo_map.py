"""
Build a topo map for a THOR env
"""
from corrsearch.experiments.domains.thor.thor import *
from corrsearch.experiments.domains.thor.visualizer import *
from corrsearch.experiments.domains.thor.tests import DummyProblem
import cv2
import time
import os
import pygame
import json
import numpy as np
from pprint import pprint

def get_clicked_pos(r, w, l):
    pos = None
    pygame.event.clear()
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
        color = (242, 212, 203, 255)
    x, y = pos
    cv2.rectangle(img, (y*r, x*r), (y*r+r, x*r+r),
                  color, -1)
    # Draw boundary
    cv2.rectangle(img, (y*r, x*r), (y*r+r, x*r+r),
                  (0, 0, 0), linewidth)

    if not unmark:
        font                   = cv2.FONT_HERSHEY_COMPLEX_SMALL
        fontScale              = 0.72
        fontColor              = (43, 13, 4)
        lineType               = 1
        imgtxt = np.full((r, r, 4), color, dtype=np.uint8)
        text_loc = (int(round(r/4)), int(round(r/1.5)))
        cv2.putText(imgtxt, str(nid), text_loc, #(y*r+r//4, x*r+r//2),
                    font, fontScale, fontColor, lineType)
        imgtxt = cv2.rotate(imgtxt, cv2.ROTATE_90_CLOCKWISE)
        img[x*r:x*r+r, y*r:y*r+r] = imgtxt
    return img

def draw_edge(img, pos1, pos2, r, thickness=2):
    x1, y1 = pos1
    x2, y2 = pos2
    cv2.line(img, (y1*r+r//2, x1*r+r//2), (y2*r+r//2, x2*r+r//2),
             (0, 0, 0, 255), thickness=thickness)
    return img

def draw_topo(img, topo_spec, grid_map, grid_size, viz):
    for nid1, nid2 in topo_spec["edges"]:
        thor_pos1 = topo_spec["nodes"][nid1]["x"], topo_spec["nodes"][nid1]["z"]
        thor_pos2 = topo_spec["nodes"][nid2]["x"], topo_spec["nodes"][nid2]["z"]
        pos1 = grid_map.to_grid_pos(*thor_pos1, grid_size=grid_size, avoid_obstacle=True)
        pos2 = grid_map.to_grid_pos(*thor_pos2, grid_size=grid_size, avoid_obstacle=True)
        img = draw_edge(img, pos1, pos2, viz._res)

    for nid in topo_spec["nodes"]:
        thor_pos = topo_spec["nodes"][nid]["x"], topo_spec["nodes"][nid]["z"]
        pos = grid_map.to_grid_pos(*thor_pos, grid_size=grid_size, avoid_obstacle=True)
        img = mark_cell(img, pos, int(nid), viz._res)

    return img


def print_options(options):
    for num, txt in sorted(options.items()):
        print(num, ":", txt)

def builder(scene_name, grid_size=0.25):
    config = {
        "scene_name": scene_name,
        "width": 400,
        "height": 400,
        "grid_size": grid_size
    }
    robot_id = 0
    controller = launch_controller(config)
    scene_info = load_scene_info(scene_name, data_path="../data")
    grid_map = convert_scene_to_grid_map(controller, scene_info, grid_size)

    # Display grid map, starts interaction
    problem = DummyProblem(robot_id)
    problem.grid_map = grid_map
    problem.objects = {}

    viz = ThorViz(problem)
    img = viz.gridworld_img()
    viz.show_img(img)

    topo_spec = {"nodes": {}, "edges": []}

    OPTIONS = {
        0: "add_node",
        1: "add_edge",
        2: "remove_node",
        3: "save",
        4: "load"
    }
    last_opt = None
    while True:
        print_options(OPTIONS)
        opt = input("Action [{}-{}]: ".format(min(OPTIONS), max(OPTIONS)))
        if len(opt) == 0:
            opt = last_opt

        try:
            opt = int(opt)
            action = OPTIONS[opt]
            last_opt = opt
        except Exception:
            print("Invalid option {}".format(opt))
            continue

        if action == "add_node":
            print("Click on window to select cell")
            x, y = get_clicked_pos(viz._res, grid_map.width, grid_map.length)
            if len(topo_spec["nodes"]) == 0:
                nid = 0
            else:
                max_nid = max(map(int, topo_spec["nodes"]))
                possnids = [possible_nid for possible_nid in np.arange(max_nid)
                            if possible_nid not in map(int, topo_spec["nodes"])]
                if len(possnids) > 0:
                    nid = min(possnids)
                else:
                    nid = len(topo_spec["nodes"])

            thor_pos = grid_map.to_thor_pos(x, y, grid_size=config["grid_size"])
            topo_spec["nodes"][str(nid)] = {"x": thor_pos[0], "z": thor_pos[1]}
            img = mark_cell(img, (x,y), nid, viz._res, viz._linewidth)
            viz.show_img(img)
            print("Added node {} to {} (aka. {})".format(nid, thor_pos, (x,y)))

        elif action == "add_edge":
            nid1 = input("Node 1: ")
            nid2 = input("Node 1: ")

            if nid1 not in topo_spec["nodes"]:
                print("Node {} is invalid".format(nid1))
            elif nid2 not in topo_spec["nodes"]:
                print("Node {} is invalid".format(nid2))
            else:
                thor_pos1 = topo_spec["nodes"][nid1]["x"], topo_spec["nodes"][nid1]["z"]
                thor_pos2 = topo_spec["nodes"][nid2]["x"], topo_spec["nodes"][nid2]["z"]
                pos1 = grid_map.to_grid_pos(*thor_pos1, grid_size=config["grid_size"])
                pos2 = grid_map.to_grid_pos(*thor_pos2, grid_size=config["grid_size"])
                topo_spec["edges"].append([nid1, nid2])

                # redraw
                img = viz.gridworld_img()
                img = draw_topo(img, topo_spec, grid_map, config["grid_size"], viz)
                viz.show_img(img)


        elif action == "save":
            default_path = "../data/topo/{}-topo-{}.json".format(scene_name, config["grid_size"])
            savepath = input("Save Path [default {}]: ".format(default_path))
            if len(savepath) == 0:
                savepath = default_path
            os.makedirs(os.path.dirname(savepath), exist_ok=True)
            with open(savepath, "w") as f:
                json.dump(topo_spec, f, sort_keys=True, indent=4)

            pygame.image.save(viz._display_surf, "../data/topo/{}-topo.png".format(scene_name))

        elif action == "load":
            default_path = "../data/topo/{}-topo.json".format(scene_name)
            loadpath = input("Load Path: [default {}]".format(default_path))
            if len(loadpath) == 0:
                loadpath = default_path
            if not os.path.exists(loadpath):
                print("Path not found")
            else:
                with open(loadpath) as f:
                    topo_spec = json.load(f)
                img = draw_topo(img, topo_spec, grid_map, config["grid_size"], viz)
                viz.show_img(img)

        elif action == "remove_node":
            removing_nid = input("Node to remove: ")

            if removing_nid not in topo_spec["nodes"]:
                print("Node {} is invalid".format(removing_nid))
                continue

            removing_nid = int(removing_nid)

            if removing_nid is not None:
                topo_spec["nodes"].pop(int(removing_nid), None)
                topo_spec["nodes"].pop(str(removing_nid), None)
                topo_spec["edges"] = [
                    edge for edge in topo_spec["edges"]
                    if removing_nid not in tuple(map(int,edge))
                ]

                # redraw
                img = viz.gridworld_img()
                img = draw_topo(img, topo_spec, grid_map, config["grid_size"], viz)
                viz.show_img(img)

            else:
                print("Misclicked. Please click on a node")



if __name__ == "__main__":
    scene_name = input("Scene name [default: FloorPlan_Train1_1]: ")
    if len(scene_name) == 0:
        scene_name = "FloorPlan_Train1_1"
    grid_size = input("Grid size [default: 0.25]: ")
    if len(grid_size) == 0:
        grid_size = 0.25
    grid_size = float(grid_size)
    builder(scene_name, grid_size=grid_size)
