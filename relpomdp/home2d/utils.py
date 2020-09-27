import math
import numpy as np
import tarfile
import shutil
from datetime import datetime as dt
import cv2
import os
import pomdp_py

# Utility functions
def euclidean_dist(p1, p2):
    return math.sqrt(sum([(a - b)** 2 for a, b in zip(p1, p2)]))

def intersect(seg1, seg2):
    """seg1 and seg2 are two line segments each represented by
    the end points (x,y). The algorithm comes from
    https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect"""
    # Represent each segment (p,p+r) where r = vector of the line segment
    seg1 = np.asarray(seg1)
    p, r = seg1[0], seg1[1]-seg1[0]
    
    seg2 = np.asarray(seg2)
    q, s = seg2[0], seg2[1]-seg2[0]

    r_cross_s = np.cross(r, s)
    if r_cross_s != 0:
        t = np.cross(q-p, s) / r_cross_s
        u = np.cross(q-p, r) / r_cross_s    
        if 0 <= t <= 1 and 0 <= u <= 1:
            # Two lines meet at point
            return True
        else:
            # Are not parallel and not intersecting
            return False
    else:
        if np.cross(q-p, r) == 0:
            # colinear
            t0 = np.dot((q - p), r) / np.dot(r, r)
            t1 = t0 + np.dot(s, r) / np.dot(r, r)
            if t0 <= 0 <= t1 or t0 <= 1 <= t1:
                # colinear and overlapping
                return True
            else:
                # colinear and disjoint
                return False
        else:
            # two lines are parallel and non intersecting
            return False

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

                
## POMDP UTILS
def objstate(obj_class, **attrs):
    return pomdp_py.ObjectState(obj_class, attrs)
