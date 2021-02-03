"""
This test identifies cases where V extended to all nonnegative vectors
is not convex.

What is happening is that I randomly generate
v1, v3
which are vectors of length 2 that represent an unnormalized belief.

I can compute corresponding b1, b2 by normalizing them.

Then, I let v2 to be a convex combination of v1 and v3
and b2 be a convex combination of b1 and b2.

I compute

V(v1), V(v2), V(v3)

and check if they satisfy x*V(v1) + (1-x)*V(v3) <= V(v2)
"""

import numpy as np
from value import extended_value, value, valueM, extended_valueM
from convhull_verification import make_dict
from tiger_problem import *
import numpy as np
import random

for i in range(10000):
    print(i)
    v1 = np.random.uniform(0, 1000, 2)
    v3 = np.random.uniform(0, 1000, 2)
    x = 0.2

    b1 = v1 / sum(v1)
    b3 = v3 / sum(v3)

    b2 = x*b1 + (1-x)*b3
    v2 = x*v1 + (1-x)*v3
    print(v2/sum(v2))
    print(b2)


    TigerTransitionModel.STATES = {TigerState("tiger-left"),
                                   TigerState("tiger-right")}
    TigerObservationModel.OBSERVATIONS = {TigerObservation("tiger-left"),
                                          TigerObservation("tiger-right")}
    TigerPolicyModel.ACTIONS = {TigerAction(s)\
                                for s in {"open-left",
                                          "open-right",
                                          "listen"}}
    agent = TigerProblem.create_instance(noise=0.15).agent
    M1 = make_dict(agent)
    states = list(M1["S"])
    assert len(states) == 2

    horizon = 3

    print("-values-")
    valv1 = extended_valueM(v1, M1, horizon=horizon)
    valv2 = extended_valueM(v2, M1, horizon=horizon)
    valv3 = extended_valueM(v3, M1, horizon=horizon)
    print(valv1)
    print(valv2)
    print(valv3)
    check = False
    if valv1*x + valv3*(1-x) > valv2:
        check = True
    elif abs(x*valv1 + (1-x)*valv3 - valv2) <= 1e-12:
        check = True
    if not check:
        # We found a failure case.
        print("FFFFFFFFFFFAILED")
        valb1 = extended_valueM(b1, M1, horizon=horizon)
        valb2 = extended_valueM(b2, M1, horizon=horizon)
        valb3 = extended_valueM(b3, M1, horizon=horizon)
        print("----- CASE -------")
        print("v#  v  V(v)")
        print("v1", v1, valv1)
        print("v3", v3, valv3)
        print("b#  b  V(b)")
        print("b1", b1, valb1)
        print("b3", b3, valb3)

        print("convex coefficient:")
        print("x = %.3f" % x)

        print("----- RESULT ------")
        print("V(v1)*x + V(v3)*(1-x) = ", valv1*x + valv3*(1-x))
        print("                V(v2) = ", valv2)
        print("V(b1)*x + V(b3)*(1-x) = ", valb1*x + valb3*(1-x))
        print("                V(b2) = ", valb2)

        break
