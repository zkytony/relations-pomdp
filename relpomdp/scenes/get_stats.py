# Obtain the frequency of object categories

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("replica_objects.csv")

apartments = {"apartment_0", "apartment_1", "apartment_2",
              "room_0", "room_1", "room_2",
              "frl_apartment_0", "frl_apartment_1", "frl_apartment_2",
              "frl_apartment_3","frl_apartment_4","frl_apartment_5"}

dapts = df.loc[df["scene_name"].isin(apartments)]


for apt in dapts["scene_name"].unique():
    d = dapts.loc[dapts["scene_name"] == apt]

    uniq_catgs = d["category"].unique()
    counts = {}
    for catg in uniq_catgs:
        objects = d.loc[d["category"] == catg]
        counts[catg] = len(objects)

    xvals = []
    yvals = []
    for catg in sorted(counts, key=counts.get):
        xvals.append(catg)
        yvals.append(counts[catg])
    plt.bar(np.arange(len(xvals)), yvals)
    plt.xticks(np.arange(len(xvals)), xvals, rotation='vertical')
    plt.title(apt)
    plt.tight_layout()
    plt.show()
    
    nuniq = len(d["category"].unique())
    print(apt, nuniq)
