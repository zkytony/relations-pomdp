import math
import numpy as np
import tarfile
import shutil
from datetime import datetime as dt
import cv2
import os
import pomdp_py
import relpomdp.oopomdp.framework as oopomdp
from relpomdp.utils_geometry import *

# Utility functions

#### File Utils ####
def save_images_and_compress(images, outdir, filename="images", img_type="png"):
    # First write the images as temporary files into the outdir
    cur_time = dt.now()
    cur_time_str = cur_time.strftime("%Y%m%d%H%M%S%f")[:-3]    
    img_save_dir = os.path.join(outdir, "tmp_imgs_%s" % cur_time_str)
    os.makedirs(img_save_dir)

    for i, img in enumerate(images):
        img = img.astype(np.float32)
        img = cv2.flip(img, 0)
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)  # rotate 90deg clockwise
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        save_path = os.path.join(img_save_dir, "tmp_img%d.%s" % (i, img_type))
        cv2.imwrite(save_path, img)

    # Then compress the image files in the outdir
    output_filepath = os.path.join(outdir, "%s.tar.gz" % filename)
    with tarfile.open(output_filepath, "w:gz") as tar:
        tar.add(img_save_dir, arcname=filename)        

    # Then remove the temporary directory
    shutil.rmtree(img_save_dir)


####### VIZ

@staticmethod
def draw_belief(img, belief, r, size, colors):
    """belief (OOBelief)"""
    radius = int(round(r / 2))

    circle_drawn = {}  # map from pose to number of times drawn

    if type(belief) == dict:
        object_beliefs = belief
    else:
        object_beliefs = belief.object_beliefs

    for objid in object_beliefs:
        if isinstance(object_beliefs[objid], dict):
            hist = object_beliefs[objid]
        else:
            if isinstance(object_beliefs[objid].random(), RobotState):
                continue
            hist = object_beliefs[objid].get_histogram()

        if objid in colors:
            color = colors[objid]
        else:
            color = (255, 165, 0)

        last_val = -1
        count = 0
        for state in reversed(sorted(hist, key=hist.get)):
            if last_val != -1:
                color = util.lighter(color, 1-hist[state]/last_val)
            if np.mean(np.array(color) / np.array([255, 255, 255])) < 0.999:
                tx, ty = state['pose']
                if (tx,ty) not in circle_drawn:
                    circle_drawn[(tx,ty)] = 0
                circle_drawn[(tx,ty)] += 1

                cv2.circle(img, (ty*r+radius,
                                 tx*r+radius), size//circle_drawn[(tx,ty)], color, thickness=-1)
                last_val = hist[state]

                count +=1
                if last_val <= 0:
                    break


# # Draw belief
# if self._last_belief is not None:
#     ObjectSearchViz.draw_belief(img, self._last_belief, r, r//3, self._colors)
