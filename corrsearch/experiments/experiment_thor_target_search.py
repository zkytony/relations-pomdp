import pomdp_py
import random
import yaml
from corrsearch.experiments.domains.thor.problem import *
from corrsearch.experiments.domains.thor.thor import *
from corrsearch.experiments.domains.thor.test_trial import *

grid_size = 0.25

case1_kitchen = {
    "scene": "FloorPlan1",
    "scene_type": "kitchen"
    "objects":
    (("Knife", dict(fov=60, max_range=0.5, truepos=0.7))
     ("Toaster", dict(rel="nearby", radius=0.75, fov=80, max_range=1.5, truepos=0.9)),
      ("Shelf", dict(rel="not_nearby", radius=1.5, fov=80, max_range=2.0, truepos=0.95)))
}

case2_kitchen = {
    "scene": "FloorPlan2",
    "scene_type": "kitchen"
    "objects":
    (("Apple", dict(fov=60, max_range=0.5, truepos=0.7))
     ("Fridge", dict(rel="nearby", radius=0.75, fov=80, max_range=2.0, truepos=0.95)),
     ("Microwave", dict(rel="not_nearby", radius=1.5, fov=80, max_range=1.5, truepos=0.9)))
}

case3_living = {
    "scene": "FloorPlan201",
    "scene_type": "living_room"
    "objects":
    (("Laptop", dict(fov=60, max_range=0.75, truepos=0.8))
     ("DiningTable", dict(rel="nearby", radius=1.0, fov=80, max_range=2.0, truepos=0.95)),
     ("DeskLamp", dict(rel="not_nearby", radius=1.25, fov=80, max_range=0.75, truepos=0.9)))
}


case4_living = {
    "scene": "FloorPlan202",
    "scene_type": "living_room"
    "objects":
    (("KeyChain", dict(fov=60, max_range=0.5, truepos=0.7))
     ("TVStand", dict(rel="nearby", radius=1.0, fov=80, max_range=1.5, truepos=0.9)),
     ("Sofa", dict(rel="not_nearby", radius=1.5, fov=80, max_range=2.0, truepos=0.95)))
}

case5_bedroom = {
    "scene": "FloorPlan301",
    "scene_type": "bedroom"
    "objects":
    (("CellPhone", dict(fov=60, max_range=0.5, truepos=0.7))
     ("Bed", dict(rel="nearby", radius=1.5, fov=80, max_range=2.0, truepos=0.95)),
     ("GarbageCan", dict(rel="not_nearby", radius=1.5, fov=80, max_range=1.5, truepos=0.9)))
}

case6_bedroom = {
    "scene": "FloorPlan302",
    "scene_type": "bedroom"
    "objects":
    (("CD", dict(fov=60, max_range=0.5, truepos=0.7))
     ("Shelf", dict(rel="nearby", radius=1.5, fov=80, max_range=1.5, truepos=0.95)),
     ("Bed", dict(rel="not_nearby", radius=1.5, fov=80, max_range=2.0, truepos=0.9)))
}

case7_bathroom = {
    "scene": "FloorPlan401",
    "scene_type": "bathroom"
    "objects":
    (("Towel", dict(fov=60, max_range=0.5, truepos=0.7))
     ("TowelHolder", dict(rel="nearby", radius=1.5, fov=80, max_range=1.25, truepos=0.8)),
     ("GarbageCan", dict(rel="not_nearby", radius=2.0, fov=80, max_range=1.5, truepos=0.9)))
}

case8_bathroom = {
    "scene": "FloorPlan402",
    "scene_type": "bathroom"
    "objects":
    (("SprayBottle", dict(fov=60, max_range=0.5, truepos=0.7))
     ("SinkBasin", dict(rel="nearby", radius=1.25, fov=80, max_range=0.75, truepos=0.8)),
     ("ShowerDoor", dict(rel="not_nearby", radius=2.25, fov=80, max_range=2.0, truepos=0.9)))
}
