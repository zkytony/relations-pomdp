import numpy as np
import random
import math
from scipy.spatial.transform import Rotation as scipyR
import scipy.stats as stats
import math
import cv2
import shutil
import tarfile
import scipy.stats
import pandas as pd
import matplotlib.pyplot as plt

def indicator(cond, epsilon=0.0):
    return 1.0 - epsilon if cond else epsilon

def euclidean_dist(p1, p2):
    return math.sqrt(sum([(a - b)** 2 for a, b in zip(p1, p2)]))

def normalize_log_prob(likelihoods):
    """Given an np.ndarray of log values, take the values out of the log space,
    and normalize them so that they sum up to 1"""
    normalized = np.exp(likelihoods -   # plus and minus the max is to prevent overflow
                        (np.log(np.sum(np.exp(likelihoods - np.max(likelihoods)))) + np.max(likelihoods)))
    return normalized

def uniform(size, ranges):
    return tuple(random.randrange(ranges[i][0], ranges[i][1])
                 for i in range(size))

def diff(rang):
    return rang[1] - rang[0]

def in_range(x, rang):
    return x >= rang[0] and x < rang[1]

def in_range_inclusive(x, rang):
    return x >= rang[0] and x <= rang[1]

def in_region(p, ranges):
    return in_range(p[0], ranges[0]) and in_range(p[1], ranges[1]) and in_range(p[2], ranges[2])

def remap(oldval, oldmin, oldmax, newmin, newmax):
    return (((oldval - oldmin) * (newmax - newmin)) / (oldmax - oldmin)) + newmin

# Printing
def json_safe(obj):
    if isinstance(obj, bool):
        return str(obj).lower()
    elif isinstance(obj, (list, tuple)):
        return [json_safe(item) for item in obj]
    elif isinstance(obj, dict):
        return {json_safe(key):json_safe(value) for key, value in obj.items()}
    else:
        return str(obj)
    return obj


# Math
def to_radians(th):
    return th*np.pi / 180

def to_rad(th):
    return th*np.pi / 180

def to_degrees(th):
    return th*180 / np.pi

def to_deg(th):
    return th*180 / np.pi

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def vec(p1, p2):
    """ vector from p1 to p2 """
    if type(p1) != np.ndarray:
        p1 = np.array(p1)
    if type(p2) != np.ndarray:
        p2 = np.array(p2)
    return p2 - p1

def proj(vec1, vec2, scalar=False):
    # Project vec1 onto vec2. Returns a vector in the direction of vec2.
    scale = np.dot(vec1, vec2) / np.linalg.norm(vec2)
    if scalar:
        return scale
    else:
        return vec2 * scale

def R_x(th):
    return np.array([
        1, 0, 0, 0,
        0, np.cos(th), -np.sin(th), 0,
        0, np.sin(th), np.cos(th), 0,
        0, 0, 0, 1
    ]).reshape(4,4)

def R_y(th):
    return np.array([
        np.cos(th), 0, np.sin(th), 0,
        0, 1, 0, 0,
        -np.sin(th), 0, np.cos(th), 0,
        0, 0, 0, 1
    ]).reshape(4,4)

def R_z(th):
    return np.array([
        np.cos(th), -np.sin(th), 0, 0,
        np.sin(th), np.cos(th), 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    ]).reshape(4,4)

def T(dx, dy, dz):
    return np.array([
        1, 0, 0, dx,
        0, 1, 0, dy,
        0, 0, 1, dz,
        0, 0, 0, 1
    ]).reshape(4,4)

def R_between(v1, v2):
    if len(v1) != 3 or len(v2) != 3:
        raise ValueError("Only applicable to 3D vectors!")
    v = np.cross(v1, v2)
    c = np.dot(v1, v2)
    s = np.linalg.norm(v)
    I = np.identity(3)

    vX = np.array([
        0, -v[2], v[1],
        v[2], 0, -v[0],
        -v[1], v[0], 0
    ]).reshape(3,3)
    R = I + vX + np.matmul(vX,vX) * ((1-c)/(s**2))
    return R

def R_euler(thx, thy, thz, affine=False):
    """
    Obtain the rotation matrix of Rz(thx) * Ry(thy) * Rx(thz); euler angles
    """
    R = scipyR.from_euler("xyz", [thx, thy, thz], degrees=True)
    if affine:
        aR = np.zeros((4,4), dtype=float)
        aR[:3,:3] = R.as_dcm()
        aR[3,3] = 1
        R = aR
    return R

def R_quat(x, y, z, w, affine=False):
    R = scipyR.from_quat([x,y,z,w])
    if affine:
        aR = np.zeros((4,4), dtype=float)
        aR[:3,:3] = R.as_dcm()
        aR[3,3] = 1
        R = aR
    return R

def R_to_euler(R):
    """
    Obtain the thx,thy,thz angles that result in the rotation matrix Rz(thx) * Ry(thy) * Rx(thz)
    Reference: http://planning.cs.uiuc.edu/node103.html
    """
    return R.as_euler('xyz', degrees=True)
    # # To prevent numerical errors, avoid super small values.
    # epsilon = 1e-9
    # matrix[abs(matrix - 0.0) < epsilon] = 0.0
    # thz = to_degrees(math.atan2(matrix[1,0], matrix[0,0]))
    # thy = to_degrees(math.atan2(-matrix[2,0], math.sqrt(matrix[2,1]**2 + matrix[2,2]**2)))
    # thx = to_degrees(math.atan2(matrix[2,1], matrix[2,2]))
    # return thx, thy, thz

def R_to_quat(R):
    return R.as_quat()

def euler_to_quat(thx, thy, thz):
    return scipyR.from_euler("xyz", [thx, thy, thz], degrees=True).as_quat()

def quat_to_euler(x, y, z, w):
    return scipyR.from_quat([x,y,z,w]).as_euler("xyz", degrees=True)

def approx_equal(v1, v2, epsilon=1e-6):
    if len(v1) != len(v2):
        return False
    for i in range(len(v1)):
        if abs(v1[i] - v2[i]) > epsilon:
            return False
    return True

def R2d(th):
    return np.array([
        np.cos(th), -np.sin(th),
        np.sin(th), np.cos(th)
    ]).reshape(2,2)


## Geometry
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

def overlap(seg1, seg2):
    """returns true if line segments seg1 and 2 are
    colinear and overlapping"""
    seg1 = np.asarray(seg1)
    p, r = seg1[0], seg1[1]-seg1[0]

    seg2 = np.asarray(seg2)
    q, s = seg2[0], seg2[1]-seg2[0]

    r_cross_s = np.cross(r, s)
    if r_cross_s == 0:
        if np.cross(q-p, r) == 0:
            # colinear
            t0 = np.dot((q - p), r) / np.dot(r, r)
            t1 = t0 + np.dot(s, r) / np.dot(r, r)
            if t0 <= 0 <= t1 or t0 <= 1 <= t1:
                # colinear and overlapping
                return True
    return False


# Others
def safe_slice(arr, start, end):
    true_start = max(0, min(len(arr)-1, start))
    true_end = max(0, min(len(arr)-1, end))
    return arr[true_start:true_end]

# Colors on terminal https://stackoverflow.com/a/287944/2893053
class bcolors:
    WHITE = '\033[97m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'

    @staticmethod
    def disable():
        bcolors.WHITE   = ''
        bcolors.CYAN    = ''
        bcolors.MAGENTA = ''
        bcolors.BLUE    = ''
        bcolors.GREEN   = ''
        bcolors.YELLOW  = ''
        bcolors.RED     = ''
        bcolors.ENDC    = ''

    @staticmethod
    def s(color, content):
        """Returns a string with color when shown on terminal.
        `color` is a constant in `bcolors` class."""
        return color + content + bcolors.ENDC

def print_info(content):
    print(bcolors.s(bcolors.BLUE, content))

def print_note(content):
    print(bcolors.s(bcolors.YELLOW, content))

def print_error(content):
    print(bcolors.s(bcolors.RED, content))

def print_warning(content):
    print(bcolors.s(bcolors.YELLOW, content))

def print_success(content):
    print(bcolors.s(bcolors.GREEN, content))

def print_info_bold(content):
    print(bcolors.BOLD + bcolors.s(bcolors.BLUE, content))

def print_note_bold(content):
    print(bcolors.BOLD + bcolors.s(bcolors.GREEN, content))

def print_error_bold(content):
    print(bcolors.BOLD + bcolors.s(bcolors.RED, content))

def print_warning_bold(content):
    print(bcolors.BOLD + bcolors.s(bcolors.YELLOW, content))

def print_success_bold(content):
    print(bcolors.BOLD + bcolors.s(bcolors.GREEN, content))
# For your convenience:
# from moos3d.util import print_info, print_error, print_warning, print_success, print_info_bold, print_error_bold, print_warning_bold, print_success_bold



# colors
def lighter(color, percent):
    '''assumes color is rgb between (0, 0, 0) and (255, 255, 255)
    If `change_alpha` is True, then the alpha will also be redueced
    by the specified amount.'''
    color = np.array(color)
    white = np.array([255, 255, 255])
    vector = white-color
    return color + vector * percent

def lighter_with_alpha(color, percent):
    '''assumes color is rgb between (0, 0, 0) and (255, 255, 255)
    If `change_alpha` is True, then the alpha will also be redueced
    by the specified amount.'''
    color = np.array(color)
    white = np.array([255, 255, 255, 255])
    vector = white-color
    return color + vector * percent

def rgb_to_hex(rgb):
    r,g,b = rgb
    return '#%02x%02x%02x' % (int(r), int(g), int(b))

def hex_to_rgb(hx):
    """hx is a string, begins with #. ASSUME len(hx)=7."""
    if len(hx) != 7:
        raise ValueError("Hex must be #------")
    hx = hx[1:]  # omit the '#'
    r = int('0x'+hx[:2], 16)
    g = int('0x'+hx[2:4], 16)
    b = int('0x'+hx[4:6], 16)
    return (r,g,b)

def inverse_color_rgb(rgb):
    r,g,b = rgb
    return (255-r, 255-g, 255-b)

def inverse_color_hex(hx):
    """hx is a string, begins with #. ASSUME len(hx)=7."""
    return inverse_color_rgb(hex_to_rgb(hx))

def random_unique_color(colors, ctype=1):
    """
    ctype=1: completely random
    ctype=2: red random
    ctype=3: blue random
    ctype=4: green random
    ctype=5: yellow random
    """
    if ctype == 1:
        color = "#%06x" % random.randint(0x444444, 0x999999)
        while color in colors:
            color = "#%06x" % random.randint(0x444444, 0x999999)
    elif ctype == 2:
        color = "#%02x0000" % random.randint(0xAA, 0xFF)
        while color in colors:
            color = "#%02x0000" % random.randint(0xAA, 0xFF)
    elif ctype == 4:  # green
        color = "#00%02x00" % random.randint(0xAA, 0xFF)
        while color in colors:
            color = "#00%02x00" % random.randint(0xAA, 0xFF)
    elif ctype == 3:  # blue
        color = "#0000%02x" % random.randint(0xAA, 0xFF)
        while color in colors:
            color = "#0000%02x" % random.randint(0xAA, 0xFF)
    elif ctype == 5:  # yellow
        h = random.randint(0xAA, 0xFF)
        color = "#%02x%02x00" % (h, h)
        while color in colors:
            h = random.randint(0xAA, 0xFF)
            color = "#%02x%02x00" % (h, h)
    else:
        raise ValueError("Unrecognized color type %s" % (str(ctype)))
    return color

# CV2, image processing
def overlay(img1, img2, opacity=1.0, pos=(0,0)):
    """
    Returns an image that is the result of overlaying img2
    on top of img1. Works for both RGB and RGBA images.

    img1 is the background image. img2 is the image to
    put on top of img1. img1 will be modified.

    Note that for opencv images, x is the column, y is the row.
    """
    # Determine the pixels that will be affected
    x_offset, y_offset = pos
    assert 0 <= x_offset < img1.shape[1],\
        "Invalid x offset (%d). Acceptable range [0,%d)" % (x_offset, img1.shape[1])
    assert 0 <= y_offset < img1.shape[0],\
        "Invalid x offset (%d). Acceptable range [0,%d)" % (y_offset, img1.shape[0])
    # img1 region
    xs1, xf1 = x_offset, min(x_offset + img2.shape[1], img1.shape[1])
    ys1, yf1 = y_offset, min(y_offset + img2.shape[0], img1.shape[0])

    xs2, xf2 = 0, min(img2.shape[1], img1.shape[1] - x_offset)
    ys2, yf2 = 0, min(img2.shape[0], img1.shape[0] - y_offset)

    if img2.shape[2] == 4:
        # Get the alpha channel of img2
        alpha = opacity * (img2[ys2:yf2, xs2:xf2, 3] / 255.0)
    else:
        alpha = opacity

    for c in range(3):
        img1[ys1:yf1, xs1:xf1, c] =\
            (alpha * img2[ys2:yf2, xs2:xf2, c]\
             + (1.0-alpha) * img1[ys1:yf1, xs1:xf1, c])
    return img1

def cv2shape(img, func, *args, alpha=1.0, **kwargs):
    """Draws cv2 shape using `func` with arguments,
    on top of given image `img` that allows transparency."""
    img_paint = img.copy()
    func(img_paint, *args, **kwargs)
    img = cv2.addWeighted(img_paint, alpha, img, 1. - alpha, 0)
    return img

# Plotting
def plot_pose(ax, pos, rot, color='b', radians=True):
    """
    pos: (x,y),
    rot: angle"""
    if not radians:
        rot = to_rad(rot)
    ax.scatter([pos[0]], [pos[1]], c=color)
    ax.arrow(pos[0], pos[1],
             0.2*math.cos(rot),  # dx
             0.2*math.sin(rot),  # dy
             width=0.005, head_width=0.05, color=color)

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


# Others
def discounted_cumulative_reward(rewards, discount_factor):
    total = 0
    d = 1.0
    for r in rewards:
        total += r*d
        d *= discount_factor
    return total

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

## Statistics
# confidence interval
def ci_normal(series, confidence_interval=0.95):
    series = np.asarray(series)
    tscore = stats.t.ppf((1 + confidence_interval)/2.0, df=len(series)-1)
    y_error = stats.sem(series)
    ci = y_error * tscore
    return ci

def mean_ci_normal(series, confidence_interval=0.95):
    ### CODE BY CLEMENT at LIS ###
    """Confidence interval for normal distribution with unknown mean and variance.

    Interpretation:
    An easy way to remember the relationship between a 95%
    confidence interval and a p-value of 0.05 is to think of the confidence interval
    as arms that "embrace" values that are consistent with the data. If the null
    value is "embraced", then it is certainly not rejected, i.e. the p-value must be
    greater than 0.05 (not statistically significant) if the null value is within
    the interval. However, if the 95% CI excludes the null value, then the null
    hypothesis has been rejected, and the p-value must be < 0.05.
    """
    series = np.asarray(series)
    # this is the "percentage point function" which is the inverse of a cdf
    # divide by 2 because we are making a two-tailed claim
    tscore = scipy.stats.t.ppf((1 + confidence_interval)/2.0, df=len(series)-1)

    y_mean = np.mean(series)
    y_error = scipy.stats.sem(series)

    half_width = y_error * tscore
    return y_mean, half_width

def perplexity(p, base=2):
    """Measures how well a probability model predicts a sample (from the same
    distribution). On Wikipedia, given a probability distribution p(x), this
    distribution has a notion of "perplexity" defined as:

        2 ^ (- sum_x p(x) * log_2 p(x))

    The exponent is also called the "entropy" of the distribution.
    These two words, "perplexity" amd "entropy" are designed
    to be vague by Shannon himself (according to Tomas L-Perez).

    The interpretation of entropy is "level of information",
    or "level of uncertainty" inherent in the distribution.
    Higher entropy indicates higher "randomness" of the
    distribution. You can actually observe the scale of

    "p(x) * log_2 p(x)" in a graph tool (e.g. Desmos).

    I made the probability distribution of x to be P(x) = 1/2 * (sin(x)+1).
    (It's not normalized, but it's still a distribution.)
    You observe that when P(x) approaches 1, the value of p(x) * log_2 p(x)
    approaches zero. The farther P(x) is away from 1, the more negative
    is the value of p(x) * log_2 p(x). You can understand it has,
    if p(x) * log_2 p(x) --> 0, then the value of x here does not
    contribute much to the uncertainty since P(x) --> 1.
    Thus, if we sum up these quantities

    "sum_x p(x) * log_2 p(x)"
    we can understand it as: the lower this sum, the more uncertainty there is.
    If you take the negation, then it becomes the higher this quantity,
    the more the uncertainty, which is the definition of entropy.

    Why log base 2? It's "for convenience and intuition" (by Shannon) In fact
    you can do other bases, like 10, or e.

    Notice how KL-divergence is defined as:

    "-sum_x p(x) * log_2 ( p(x) / q(x) )"

    The only difference is there's another distribution q(x). It measures
    how "different" two distributions are. KL divergence of 0 means identical.

    How do you use perplexity to compare two distributions? You compute the
    perplexity of both.

    Also refer to: https://www.cs.rochester.edu/u/james/CSC248/Lec6.pdf

    Parameters:
        p: A sequence of probabilities
    """
    H = scipy.stats.entropy(p, base=base)
    return base**H

def kl_divergence(p, q, base=2):
    return scipy.stats.entropy(p, q, base=base)

def normal_pdf_2d(point, variance, domain, normalize=True):
    """
    returns a dictionary that maps a value in domain to a probability
    such that the probability distribution is a 2d gaussian with mean
    at the given point and given variance.
    """
    dist = {}
    total_prob = 0.0
    for val in domain:
        prob = scipy.stats.multivariate_normal.pdf(np.array(val),
                                                   np.array(point),
                                                   np.array(variance))
        dist[val] = prob
        total_prob += prob
    if normalize:
        for val in dist:
            dist[val] /= total_prob
    return dist

def dists_to_seqs(dists, avoid_zero=True):
    """Convert dictionary distributions to seqs (lists) such
    that the elements at the same index in the seqs correspond
    to the same key in the dictionary"""
    seqs = [[] for i in range(len(dists))]
    vals = []
    d0 = dists[0]
    for val in d0:
        for i, di in enumerate(dists):
            if val not in di:
                raise ValueError("Value %s is in one distribution but not another" % (str(val)))
            if avoid_zero:
                prob = max(1e-12, di[val])
            else:
                prob = di[val]
            seqs[i].append(prob)
        vals.append(val)
    return seqs, vals

def compute_mean_ci(results):
    """Given `results`, a dictionary mapping "result_type" to a list of values
    for this result_type, compute the mean and confidence intervals for each
    of the result type. It will add a __summary__ key to the given dictionary.x"""
    results["__summary__"] = {}
    for restype in results:
        if restype.startswith("__"):
            continue
        mean, ci = mean_ci_normal(results[restype], confidence_interval=0.95)
        results["__summary__"][restype] = {
            "mean": mean,
            "ci-95": ci,
            "size": len(results[restype])
        }
    return results

def entropy(p, base=2):
    """
    Parameters:
        p: A sequence of probabilities
    """
    return scipy.stats.entropy(p, base=base)
