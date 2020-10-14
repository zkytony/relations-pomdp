import pomdp_py
import os
import pickle
import pandas as pd
import seaborn as sns
import copy
import matplotlib.pyplot as plt
from test_mdp import test_mdp
from test_pomdp import test_pomdp
from test_pomdp_nk import test_pomdp_nk

def cum_disc(rewards, disc_factor):
    disc = 1.0
    res = 0.0
    for r in rewards:
        res += disc*r
        disc *= disc_factor
    return res

def main(map_dir="test_maps",
         nsteps=50, discount_factor=0.95):
    results = []  # rows
    try:
        for filename in os.listdir(map_dir):
            size = filename.split("_")[0].split("-")[0]
            with open(os.path.join(map_dir, filename), "rb") as f:
                env = pickle.load(f)
            rewards_mdp = test_mdp(copy.deepcopy(env),
                                   nsteps=nsteps, discount_factor=discount_factor)
            rewards_pomdp = test_pomdp(copy.deepcopy(env),
                                       nsteps=nsteps, discount_factor=discount_factor)
            rewards_pomdp_nk = test_pomdp_nk(copy.deepcopy(env),
                                             nsteps=nsteps, discount_factor=discount_factor)
            results.append((size, "mdp", cum_disc(rewards_mdp, discount_factor)))
            results.append((size, "pomdp", cum_disc(rewards_pomdp, discount_factor)))
            results.append((size, "pomdp_nk", cum_disc(rewards_pomdp_nk, discount_factor)))
    finally:
        df = pd.DataFrame(results, columns=["size", "method", "cum_disc"])
        df.to_csv("cum_disc-%d-%.2f.csv" % (nsteps, discount_factor))

if __name__ == "__main__":
    main(map_dir="small_test_maps",
         nsteps=2, discount_factor=0.95)


# sns.barplot(x="method", y="cum_disc", hue="size", data=df)
# import pdb; pdb.set_trace()
# plt.savefig("cum_disc-%d-%.2f.png" % (nsteps, discount_factor))
