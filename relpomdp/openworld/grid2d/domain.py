from relpomdp.openworld.common import *
from metric_map import MetricMap2d

# Domain-level configurations

MIN_CUPBOARD_LENGTH = 1
MAX_CUPBOARD_LENGTH = 4

MIN_TABLE_LENGTH = 2
MAX_TABLE_LENGTH = 8

TABLE_COLORS = {"red", "orange", "yellow"}

MAP_NAME = "map2"
MAP = MetricMap2d(MAP_NAME)

# Define classes, attributes and their domains
CLASSES = {
    "Cupboard": {
        "Location": (Vec2d, MAP),
        "Dimension": (Vec2d,
                      Ranges((MIN_CUPBOARD_LENGTH, MAX_CUPBOARD_LENGTH),
                             (MIN_CUPBOARD_LENGTH, MAX_CUPBOARD_LENGTH))),
        "Facing": (Real, Ranges((0, 360.))),  # angle
        "DoorAngle": (Real, Ranges((0., 90.))),
    },

    "Table": {
        "Location": (Vec2d, MAP),
        "Dimension": (Vec2d,
                      Ranges((MIN_TABLE_LENGTH, MAX_TABLE_LENGTH),
                             (MIN_TABLE_LENGTH, MAX_TABLE_LENGTH))),
        "Color": (Catg,
                  Ranges(TABLE_COLORS))
    },

    "Mail": {
        "Location": (Vec2d, MAP),
        "Dimension": (Vec2d, Ranges((1, 1), (1, 1))),
        "Opened": (Bool, Ranges({True, False}))
    },

    "Plate": {
        "Location": (Vec2d, MAP),
        "Radius": (Real, Ranges((0,1)))
    }
}


# Define instances of objects
OBJECTS = {
    1: ("Cupboard",
        ("Location", (0, 0)),
        ("Dimension", (4, 3)),
        ("Facing", 0.),
        ("DoorAngle", 45.)),
    2: ("Table",
        ("Location", (8, 9)),
        ("Dimension", (6, 3)),
        ("Color", "red")),
    3: ("Mail",
        ("Location", (6, 10)),
        ("Dimension", (1, 1))),
    4: ("Plate",
        ("Location", (8, 9)),
        ("Radius", 0.5)),
    5: ("Plate",
        ("Location", (10, 10)),
        ("Radius", 0.5))
}


