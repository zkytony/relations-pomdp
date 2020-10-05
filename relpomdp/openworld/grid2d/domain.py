import 

# Domain-level configurations

MIN_CUPBOARD_LENGTH = 1
MAX_CUPBOARD_LENGTH = 4

MIN_TABLE_LENGTH = 2
MAX_TABLE_LENGTH = 8

TABLE_COLORS = {"red", "orange", "yellow"}

MAP_NAME = "map1"

# Define classes, attributes and their domains
CLASSES = {
    "Cupboard": {
        "Location": (Vec2d, MetricMap(MAP_NAME)),
        "Dimension": (Vec2d,
                      Ranges((MIN_CUPBOARD_LENGTH, MAX_CUPBOARD_LENGTH),
                             (MIN_CUPBOARD_LENGTH, MAX_CUPBOARD_LENGTH))),
        "Facing": (Vec2d, Ranges("real", "real")),
        "DoorAngle": (Real, Ranges((0., 90.))),
    },

    "Table": {
        "Location": (Vec2d, MetricMap(MAP_NAME)),
        "Dimension": (Vec2d,
                      Ranges((MIN_TABLE_LENGTH, MAX_TABLE_LENGTH),
                             (MIN_TABLE_LENGTH, MAX_TABLE_LENGTH))),
        "Color": (Catg,
                  Ranges(TABLE_COLORS))
    },

    "Mail": {
        "Location": (Vec2d, MetricMap(MAP_NAME)),
        "Dimension": (Vec2d, Ranges((1, 1), (1, 1))),
        "Opened": (Bool, Ranges({True, False}))
    },

    "Plate": {
        "Location": (Vec2d, MetricMap(MAP_NAME)),
        "Radius": (Real, Ranges((1,1)))
    }
}
