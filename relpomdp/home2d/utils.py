import math
import numpy as np
import tarfile
import shutil
from datetime import datetime as dt
import cv2
import os

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

