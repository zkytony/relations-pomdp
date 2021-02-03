# Different from Jupyter Notebook version, this
# one is more for creating material for the analysis
# that can be shown in other medium.

import pomdp_py
import os, sys
from hallway_xy import *
from hallway_xy import main as run_domain_xy


def with_and_without(setting, spatial_corr, range_x=0, range_y=1):
    """Run agent in `setting` with SARSOP policy computed
    for the case with lookY in the action space and for
    the case without lookY in the action space."""
    print("\nComparing value between with and without lookY")
    # spatial_corr = "spatially_close"
    # setting = "YX.R."
    policy_NOlookY, _, init_belief1, _\
        = run_domain_xy(solver="sarsop", setting=setting,
                        spatial_corr_func=eval(spatial_corr),
                        range_x=range_x, range_y=range_y,
                        using_jupyter=False,
                        savedir=os.path.join("examples", spatial_corr, setting, "sarsop_NOlookY"),
                        actions=[HSAction("left"), HSAction("right"), HSAction("lookX"),
                                 HSAction("Declare")])

    policy_HASlookY, _, init_belief2, _\
        = run_domain_xy(solver="sarsop", setting=setting,
                        spatial_corr_func=eval(spatial_corr),
                        range_x=range_x, range_y=range_y,
                        using_jupyter=False,
                        savedir=os.path.join("examples", spatial_corr, setting, "sarsop_HASlookY"),
                        actions=[HSAction("left"), HSAction("right"), HSAction("lookX"),
                                 HSAction("lookY"), HSAction("Declare")])
    assert init_belief1 == init_belief2
    print("Value at initial belief WITH lookY: {}".format(policy_HASlookY.value(init_belief1)))
    print("Value at initial belief WITHOUT lookY: {}".format(policy_NOlookY.value(init_belief1)))


def test_hallway_xy():
    range_x, range_y = 0, 1

    print("\nCreating Spatially Exact example")

    spatial_corr = "spatially_exact"
    setting = "X.YR"

    print("------ SARSOP ------")
    run_domain_xy(solver="sarsop", setting=setting,
               spatial_corr_func=eval(spatial_corr),
               range_x=range_x, range_y=range_y,
               using_jupyter=False,
               savedir=os.path.join("examples", spatial_corr, setting, "sarsop"))

    print("------ VI ------")
    run_domain_xy(solver="vi", setting=setting,
               spatial_corr_func=eval(spatial_corr),
               range_x=range_x, range_y=range_y,
               using_jupyter=False, vi_options=['-horizon', 7],
               savedir=os.path.join("examples", spatial_corr, setting, "vi"))

    print("------ POUCT ------")
    run_domain_xy(solver="pouct", setting=setting,
                  spatial_corr_func=eval(spatial_corr),
                  range_x=range_x, range_y=range_y,
                  using_jupyter=False,
                  savedir=os.path.join("examples", spatial_corr, setting, "pouct"))
    with_and_without("X.YR", "spatially_exact", range_x=range_x, range_y=range_y)

    print("\nCreating Spatially Apart Example")
    with_and_without("XR.Y", "spatially_apart", range_x=range_x, range_y=range_y)
    with_and_without("X.YR", "spatially_apart", range_x=range_x, range_y=range_y)

    print("\nCreating Spatially Close Example")
    with_and_without("XY.R.", "spatially_close", range_x=range_x, range_y=range_y)
    with_and_without(".RXY", "spatially_close", range_x=range_x, range_y=range_y)

    print("\nCreating Spatially Close Example II")
    with_and_without("XY..R", "spatially_close", range_x=range_x, range_y=range_y)

    print("\nCreating Spatially Independent Example")
    with_and_without("X.YR", "spatially_independent", range_x=range_x, range_y=range_y)



if __name__ == "__main__":
    test_hallway_xy()
