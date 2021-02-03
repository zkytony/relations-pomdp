import pomdp_py
from pomdp_py.algorithms.value_function import belief_update, value, qvalue, belief_observation_model
from value import valueM
from tiger_problem import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def make_dict(agent, gamma=0.95):
    T = agent.transition_model
    O = agent.observation_model
    S = list(T.get_all_states())
    Z = list(O.get_all_observations())
    A = list(agent.policy_model.get_all_actions())
    R = agent.reward_model
    b0 = agent.belief
    return dict(b0=b0, S=S, A=A, Z=Z, T=T, O=O, R=R, gamma=gamma)

# CHANGE THESE PARAMETERS.
# Set them to 0.4, 0.9 for a counter example to the convex hull conjecture in 3-state tiger
noise1=1e-9
noise2=0.111

def test_two_states():
    # Two-State
    TigerTransitionModel.STATES = {TigerState("tiger-left"),
                                   TigerState("tiger-right")}
    TigerObservationModel.OBSERVATIONS = {TigerObservation("tiger-left"),
                                          TigerObservation("tiger-right")}
    TigerPolicyModel.ACTIONS = {TigerAction(s)\
                                for s in {"open-left",
                                          "open-right",
                                          "listen"}}
    M1 = make_dict(TigerProblem.create_instance(noise=noise1).agent)
    M2 = make_dict(TigerProblem.create_instance(noise=noise2).agent)
    # M1["b0"] = pomdp_py.Histogram({TigerState("tiger-left"):0.23})
    data = [[M1["b0"][TigerState("tiger-left")], 0.0, "b0", "M1-{}".format(noise1)],
            [M2["b0"][TigerState("tiger-left")], 0.0, "b0", "M2-{}".format(noise2)]]
    for z in M1["Z"]:
        b_a1_z = belief_update(M1["b0"], TigerAction("listen"), z, M1["T"], M1["O"])
        b_a2_z = belief_update(M2["b0"], TigerAction("listen"), z, M2["T"], M2["O"])
        data.append([b_a1_z[TigerState("tiger-left")], 0.0, "z_%s" % z, "M1-{}".format(noise1)])
        data.append([b_a2_z[TigerState("tiger-left")], 0.0, "z_%s" % z, "M2-{}".format(noise2)])

    print("M1-{}".format(noise1), valueM(M1["b0"], M1, horizon=3))
    print("M2-{}".format(noise2), valueM(M2["b0"], M2, horizon=3))
    df = pd.DataFrame(data,
                      columns=["b(s1)", "dummy", "b_name", "model"])
    sns.scatterplot(x="b(s1)", y="dummy", hue="model", data=df)
    plt.scatter(M1["b0"][TigerState("tiger-left")], [0.0],
                color="black")
    plt.legend(loc="upper right")
    plt.show()

def test_three_states():
    # Three-State
    TigerTransitionModel.STATES = {TigerState("tiger-left"),
                                   TigerState("tiger-middle"),
                                   TigerState("tiger-right")}
    TigerObservationModel.OBSERVATIONS = {TigerObservation("tiger-left"),
                                          TigerObservation("tiger-middle"),
                                          TigerObservation("tiger-right")}
    TigerPolicyModel.ACTIONS = {TigerAction(s)\
                                for s in {"open-left",
                                          "open-right",
                                          "open-middle",
                                          "listen"}}
    M1 = make_dict(TigerProblem.create_instance(noise=noise1).agent)
    M2 = make_dict(TigerProblem.create_instance(noise=noise2).agent)
    M1["b0"] = pomdp_py.Histogram({TigerState("tiger-left"):0.01,
                                   TigerState("tiger-middle"):0.98,
                                   TigerState("tiger-right"):0.01})
    data = [[M1["b0"][TigerState("tiger-left")], M1["b0"][TigerState("tiger-right")], "b0", "M1-{}".format(noise1)],
            [M2["b0"][TigerState("tiger-left")], M2["b0"][TigerState("tiger-right")], "b0", "M2-{}".format(noise2)]]

    OP_a1 = []
    OP_a2 = []
    for z in M1["Z"]:
        b_a1_z = belief_update(M1["b0"], TigerAction("listen"), z, M1["T"], M1["O"])
        b_a2_z = belief_update(M2["b0"], TigerAction("listen"), z, M2["T"], M2["O"])
        OP_a1.append([b_a1_z[TigerState("tiger-left")], b_a1_z[TigerState("tiger-right")]])
        OP_a2.append([b_a2_z[TigerState("tiger-left")], b_a2_z[TigerState("tiger-right")]])
        data.append([OP_a1[-1][0], OP_a1[-1][1], "z_%s" % z, "M1-{}".format(noise1)])
        data.append([OP_a2[-1][0], OP_a2[-1][1], "z_%s" % z, "M2-{}".format(noise2)])

    # Report
    print("Models and noises:")
    print("* M1: {}".format(noise1))
    print("* M2: {}".format(noise2))
    print("Value:")
    vM1 = valueM(M1["b0"], M1, horizon=3)
    vM2 = valueM(M2["b0"], M2, horizon=3)
    print("* V_M1(b0) =", vM1)
    print("* V_M2(b0) =", vM2)
    if abs(vM1 - vM2) < 1e-12:
        print("vM1 = vM2")
    elif vM1 > vM2:
        print("vM1 > vM2")
    else:
        print("vM1 < vM2")

    # Plotting
    df = pd.DataFrame(data,
                      columns=["b(s1)", "b(s2)", "b_name", "model"])
    ax = sns.scatterplot(x="b(s1)", y="b(s2)", hue="model", data=df, s=50)
    # For each point, add a text
    for line in range(0, df.shape[0]):
        ann = "{}\n{:.2f}\n{:.2f}".format(df["b_name"][line], df["b(s1)"][line], df["b(s2)"][line])
        ax.text(df["b(s1)"][line], df["b(s2)"][line], ann)
    # Connect the dots
    plt.plot([OP_a1[i][0] for i in range(len(OP_a1))] + [OP_a1[0][0]],
             [OP_a1[i][1] for i in range(len(OP_a1))] + [OP_a1[0][1]],
             "--", linewidth=1)
    plt.plot([OP_a2[i][0] for i in range(len(OP_a2))] + [OP_a2[0][0]],
             [OP_a2[i][1] for i in range(len(OP_a2))] + [OP_a2[0][1]],
             "--", linewidth=1)

    plt.scatter(M1["b0"][TigerState("tiger-left")], M1["b0"][TigerState("tiger-right")],
                color="black")
    plt.legend(loc="upper right")
    plt.show()

if __name__ == "__main__":
    # print("---Two-state Tiger---")
    # test_two_states()
    print("---Three-state Tiger---")
    test_three_states()
