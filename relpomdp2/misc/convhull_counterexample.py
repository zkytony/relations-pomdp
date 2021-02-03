import numpy as np
from value import extended_value, value, valueM, extended_valueM,\
    belief_update
from convhull_verification import make_dict
from tiger_problem import *
from scipy.spatial import Delaunay
import numpy as np
import random

noise1=0.4
noise2=0.3

def to_barr(b, S):
    return np.array([b[s] for s in S])

def in_convex_hull(points1, points2):
    """Returns True if all points in `points1` lie
    in the convex hull formed by `points2`"""
    # try:
    #     hull = Delaunay(points2, qhull_options='Pp')
    #     return all(hull.find_simplex(points1) >= 0)
    # except Exception as ex:
    #     print(ex)
    #     return True

    # EXPLOITING THE FACT THAT POINTS ARE 2D
    if len(points1[0]) == 2:
        pp2 = []
        for p in points2:
            # if not (p[0] + p[1] - 1.0) <= 1e-12:
            #     import pdb; pdb.set_trace()
            pp2.append(p[0])
        p2min = min(pp2)
        p2max = max(pp2)
        inhull = True
        for p in points1:
            if np.isnan(p[0]):
                return False
            if not (p[0] >= p2min and p[1] <= p2max):
                inhull = False
                return inhull
    else:
        try:
            hull = Delaunay(points2, qhull_options='Pp')
            return all(hull.find_simplex(points1) >= 0)
        except Exception as ex:
            print(ex)
            return False

def test_two_states():
    # Two-State
    s1 = TigerState("tiger-left")
    s2 = TigerState("tiger-right")
    z1 = TigerObservation("tiger-left")
    z2 = TigerObservation("tiger-right")
    TigerTransitionModel.STATES = {s1, s2}
    TigerObservationModel.OBSERVATIONS = {z1, z2}
    TigerPolicyModel.ACTIONS = {TigerAction(s)\
                                for s in {"open-left",
                                          "open-right",
                                          "listen"}}

    for noise1 in np.linspace(0.0, 1.0, num=10):
        for noise2 in np.linspace(0.0, 1.0, num=10):
            print("---- Noises: %.3f  %.3f ----" % (noise1, noise2))
            M1 = make_dict(TigerProblem.create_instance(noise=noise1).agent)
            M2 = make_dict(TigerProblem.create_instance(noise=noise2).agent)
            horizon = 3

            for bs1 in np.linspace(0.01, 0.99, num=20):
                print("b(s1) = %.3f" % bs1)
                b0 = {s1: bs1, s2: 1.0 - bs1}

                OP_M1 = [to_barr(belief_update(b0, TigerAction("listen"),
                                               z, M1["T"], M1["O"]), M1["S"])
                         for z in M1["Z"]]
                # OP_M1.extend(OP_M1)
                OP_M2 = [to_barr(belief_update(b0, TigerAction("listen"),
                                               z, M2["T"], M2["O"]), M1["S"])
                         for z in M1["Z"]]
                # OP_M2.extend(OP_M2)

                if in_convex_hull(OP_M1, OP_M2):
                    # M2 is more informative than M1
                    # Find a case where V*M1(b) > V*M2(b) --- just check if this case works
                    V_M1 = valueM(b0, M1, horizon=horizon)
                    V_M2 = valueM(b0, M2, horizon=horizon)
                    if V_M1 > V_M2 and abs(V_M1 - V_M2) > 1e-12:
                        print("!!!!!!  W E  H A V E  A  C A S E  !!!!!!")
                        print("b0: ", b0)
                        print("noise1: ", noise1)
                        print("noise2: ", noise2)
                        print("V_M1: ", V_M1)
                        print("V_M2: ", V_M2)
                        return



def test_three_states():
    # Three-State
    s1 = TigerState("tiger-left")
    s2 = TigerState("tiger-right")
    s3 = TigerState("tiger-middle")
    z1 = TigerObservation("tiger-left")
    z2 = TigerObservation("tiger-right")
    z3 = TigerObservation("tiger-middle")
    TigerTransitionModel.STATES = {s1, s2, s3}
    TigerObservationModel.OBSERVATIONS = {z1, z2, z3}
    TigerPolicyModel.ACTIONS = {TigerAction(s)\
                                for s in {"open-left",
                                          "open-right",
                                          "open-middle",
                                          "listen"}}

    for noise1 in np.linspace(1e-9, 1.0, num=10):
        for noise2 in np.linspace(1e-9, 1.0, num=10):
            print("---- Noises: %.3f  %.3f ----" % (noise1, noise2))
            M1 = make_dict(TigerProblem.create_instance(noise=noise1).agent)
            M2 = make_dict(TigerProblem.create_instance(noise=noise2).agent)
            horizon = 3

            for bs1 in np.linspace(0.01, 0.99, num=10):
                for bs2 in np.linspace(0.01, 0.99, num=10):
                    print("b(s1) = %.3f" % bs1)
                    b0 = {s1: bs1, s2: bs2, s3: 1.0 - bs1 - bs2}

                    OP_M1 = [to_barr(belief_update(b0, TigerAction("listen"),
                                                   z, M1["T"], M1["O"]), M1["S"])
                             for z in M1["Z"]]
                    OP_M1.extend(OP_M1)
                    OP_M2 = [to_barr(belief_update(b0, TigerAction("listen"),
                                                   z, M2["T"], M2["O"]), M1["S"])
                             for z in M1["Z"]]
                    OP_M2.extend(OP_M2)

                    if in_convex_hull(OP_M1, OP_M2):
                        # M2 is more informative than M1
                        # Find a case where V*M1(b) > V*M2(b) --- just check if this case works
                        print(b0)
                        V_M1 = valueM(b0, M1, horizon=horizon)
                        print(b0)
                        V_M2 = valueM(b0, M2, horizon=horizon)
                        print(b0)
                        if V_M1 > V_M2 and abs(V_M1 - V_M2) > 1e-12:
                            print("!!!!!!  W E  H A V E  A  C A S E  !!!!!!")
                            print("b0: ", b0)
                            print("noise1: ", noise1)
                            print("noise2: ", noise2)
                            print("V_M1: ", V_M1)
                            print("V_M2: ", V_M2)
                            return

if __name__ == "__main__":
    # test_two_states()
    test_three_states()
