from relpomdp.openworld.visualizer import WorldViz
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import webcolors
import math
import cv2

RESOURCE_PATH = "./resources"

# For continuous domain to work you need motion planning

def draw_rect(img, x, y, w, l, res, color_fill,
              color_boundary=(0,0,0,255), thickness=1):
    # Body
    img = cv2.rectangle(img,
                        (int(round(y*res)), int(round(x*res))),
                        (int(round((y+l)*res)), int(round((x+w)*res))),
                        color=color_fill, thickness=-1)

    # Black boundary
    img = cv2.rectangle(img,
                        (int(round(y*res)), int(round(x*res))),
                        (int(round((y+l)*res)), int(round((x+w)*res))),
                        color=color_boundary, thickness=thickness)
    return img


class CupboardViz:
    @classmethod
    def update(cls, objstate, img, res):
        # Draw the body of the cupboard
        x, y = objstate["Location"].value
        w, l = objstate["Dimension"].value
        opened = objstate["Opened"].value

        # We simply represent whether the cupboard is open
        # using a change in color. TOTHINK: Having a door is super complex.
        # Having a continous open door is super complex. Motion planning?
        if opened:
            color=(200, 200, 200, 255)
        else:
            color=(128, 128, 128, 255)
        img = draw_rect(img, x, y, w, l, res, color)
        return img


class TableViz:
    @classmethod
    def update(cls, objstate, img, res):
        # Draw the body of the table
        x, y = objstate["Location"].value
        w, l = objstate["Dimension"].value
        color = tuple(webcolors.name_to_rgb(objstate["Color"].value)) + (255,)
        img = draw_rect(img, x, y, w, l, res, color)        
        return img


class MailViz:
    @classmethod
    def update(cls, objstate, img, res):
        # Draw the body of the table
        x, y = objstate["Location"].value
        w, l = objstate["Dimension"].value
        color = (255, 244, 137, 100)
        img = draw_rect(img, x+(w*0.3/2), y+(l*0.3/2),
                        w*0.7, l*0.7, res, color,
                        color_boundary=(239, 186, 108, 255))
        return img    


class PlateViz:
    @classmethod
    def update(cls, objstate, img, res):
        # Draw the body of the table
        x, y = objstate["Location"].value
        radius = int(round(objstate["Radius"].value * res))
        shift = int(round(res / 2))
        color = (230, 249, 231, 128)
        color_boundary = (192, 222, 195, 255)
        img = cv2.circle(img, (y*res+shift, x*res+shift), radius, color, thickness=-1)
        img = cv2.circle(img, (y*res+shift, x*res+shift), radius+1, color_boundary, thickness=1)
        img = cv2.circle(img, (y*res+shift, x*res+shift), (radius)//2, color_boundary, thickness=1)        
        return img


class RobotViz:
    @classmethod
    def update(cls, objstate, img, res):
        # Draw the body
        x, y = objstate["Location"].value
        th = math.radians(objstate["Orientation"].value)
        color = (100, 149, 230, 255)
        size = res
        radius = int(round(size/2))
        img = cv2.circle(img,
                         (y*res+radius, x*res+radius),
                         radius,
                         color,
                         thickness=-1)

        color2 = (200, 209, 230, 255)
        extension = radius
        ey, ex = (y*res+radius + int(round(extension*math.sin(th))),
                  x*res+radius + int(round(extension*math.cos(th))))        
        img = cv2.circle(img, (ey, ex), radius//2, color2,
                         thickness=-1)        
        return img                
    

class GridWorldViz(WorldViz):
    """Visualize the world using matplotlib"""

    def __init__(self, metric_map, res=30):

        self.classviz = {
            "Cupboard": CupboardViz,
            "Table": TableViz,
            "Mail": MailViz,
            "Plate": PlateViz,
            "Robot": RobotViz
        }
        self.zorder = {"Table": 1,
                       "Cupboard": 2,
                       "Plate": 4,
                       "Mail": 5,
                       "Robot": 6}

        self.res = res
        self._map = metric_map
        self._img = self._make_gridworld_image(res)
        self.fig, self.ax = plt.subplots(figsize=(10,10))

    def _make_gridworld_image(self, r):
        # Preparing 2d array
        w, l = self._map.width, self._map.length
        
        # Creating image (RGBA)
        img = np.full((w*r,l*r,4), 255, dtype=np.int32)
        for x in range(w):
            for y in range(l):
                # Draw free space
                if (x,y) in self._map.obstacle_poses:
                    cv2.rectangle(img, (y*r, x*r), (y*r+r, x*r+r),
                                  (0, 0, 0, 255), -1)
                else:
                    cv2.rectangle(img, (y*r, x*r), (y*r+r, x*r+r),
                                  (255, 255, 255, 255), -1)
                # Draw boundary
                cv2.rectangle(img, (y*r, x*r), (y*r+r, x*r+r),
                              (0, 0, 0, 255), 1, 8)
        return img

    def update(self, objstates):
        img = self._img
        for objid in sorted(objstates,
                            key=lambda objid: self.zorder[objstates[objid].objclass]):
            objstate = objstates[objid]
            if objstate.objclass in self.classviz:
                img = self.classviz[objstate.objclass].update(objstate, img, self.res)
        img = np.swapaxes(img, 0, 1)
        self.ax.imshow(img, interpolation='none')
        self.ax.set_aspect("equal")
