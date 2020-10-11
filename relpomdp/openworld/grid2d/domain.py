from relpomdp.openworld.common import *
from metric_map import MetricMap2d

# Domain-level configurations

MIN_CUPBOARD_LENGTH = 1
MAX_CUPBOARD_LENGTH = 4

MIN_TABLE_LENGTH = 2
MAX_TABLE_LENGTH = 8

TABLE_COLORS = {"antiquewhite", "burlywood", "darkgray"}

MAP_NAME = "map2"
MAP = MetricMap2d(MAP_NAME)

# Define classes, attributes and their domains
CLASSES = {
    "Cupboard": {
        "Location": (Vec2d, MAP),
        "Dimension": (Vec2d,
                      Ranges((MIN_CUPBOARD_LENGTH, MAX_CUPBOARD_LENGTH),
                             (MIN_CUPBOARD_LENGTH, MAX_CUPBOARD_LENGTH))),
        "Opened": (Bool, Ranges({True, False}))
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
    },
    "Robot": {
        "Location": (Vec2d, MAP),
        "Orientation": (Real, Ranges((0.,360.)))
    }
}


# Define instances of objects
OBJECTS = {
    100: ("Cupboard",
          ("Location", (0, 5)),
          ("Dimension", (2, 2)),
          ("Opened", False)),
    101: ("Cupboard",
          ("Location", (0, 7)),
          ("Dimension", (2, 2)),
          ("Opened", True)),
    102: ("Cupboard",
          ("Location", (13, 0)),
          ("Dimension", (2, 2)),
          ("Opened", False)),
    103: ("Cupboard",
          ("Location", (15, 0)),
          ("Dimension", (2, 2)),
          ("Opened", False)),        
    200: ("Table",
          ("Location", (5, 0)),
          ("Dimension", (6, 3)),
          ("Color", "antiquewhite")),
    300: ("Mail",
          ("Location", (0, 6)),
          ("Dimension", (1, 1)),
          ("Opened", False)),
    400: ("Plate",
          ("Location", (5, 0)),
          ("Radius", 0.5)),
    401: ("Plate",
          ("Location", (8, 2)),
          ("Radius", 0.5)),
    1001: ("Robot",
          ("Location", (4, 5)),
          ("Orientation", 0.))
}
